"""Per-phase GPU energy from synchronized NVML cumulative-counter differences.

PP3 uses start/end counter reads only. CounterWindowAllocator is not connected
to the training runtime. Texture retains its separate measurement implementation.
"""
import math
import time
import torch


ENERGY_MEASUREMENT_VERSION = 'nvml_total_phase_v1'


class TotalEnergyCounter:
    """Synchronized cumulative mJ readings; never fall back to power estimates.

    A missing value means unsupported/unavailable telemetry. Equal consecutive
    readings remain valid counter snapshots, not claims of zero window energy.
    """

    def __init__(self, device):
        self.device = device
        self.nv = None
        self.previous = None
        self.reason = 'CPU run: GPU energy unavailable'
        if device.type != 'cuda':
            return
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            if not uuid.startswith(('GPU-', 'MIG-')):
                uuid = 'GPU-' + uuid
            self.handle = nv.nvmlDeviceGetHandleByUUID(uuid)
            self.reason = ''
        except Exception as error:
            self.reason = f'{type(error).__name__}: {error}'

    def read(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        if self.reason or self.nv is None:
            return time.perf_counter(), None
        try:
            value = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            if not isinstance(value, int) or value < 0:
                raise ValueError('Invalid GPU cumulative energy counter')
            if self.previous is not None and value < self.previous:
                raise ValueError('GPU energy counter reset')
            self.previous = value
            return time.perf_counter(), value
        except Exception as error:
            self.reason = f'{type(error).__name__}: {error}'
            return time.perf_counter(), None

    def close(self):
        if self.nv is not None:
            nv, self.nv = self.nv, None
            nv.nvmlShutdown()


class CounterWindowAllocator:
    """Allocate contiguous counter windows to epochs by their elapsed time.

    Callers supply synchronized (seconds, mJ) boundaries. Short epochs are
    grouped until min_seconds; long epochs are kept separate. A final partial
    group is allocated only if its counter advanced, never filled with zero.
    No assumptions about which operations fall between boundaries are made here.
    """

    def __init__(self, min_seconds=1.0):
        if not math.isfinite(min_seconds) or min_seconds <= 0:
            raise ValueError('Minimum energy window must be finite and positive')
        self.min_seconds = min_seconds
        self.pending = []
        self.allocations = {}

    def add(self, key, start, end):
        if key in self.allocations or any(item[0] == key for item in self.pending):
            raise ValueError('Duplicate energy window key')
        seconds = end[0] - start[0]
        if seconds <= 0:
            raise ValueError('Energy window duration must be positive')
        if seconds >= self.min_seconds:
            # Do not absorb a long epoch into an earlier short-epoch group.
            self.flush()
            self.pending.append((key, start, end))
            self.flush()
        else:
            self.pending.append((key, start, end))
            span = end[0] - self.pending[0][1][0]
            initial = self.pending[0][1][1]
            valid = initial is not None and end[1] is not None and end[1] > initial
            if span >= self.min_seconds and valid:
                self.flush()
        return self.allocations

    def flush(self):
        if not self.pending:
            return self.allocations
        initial, final = self.pending[0][1][1], self.pending[-1][2][1]
        values = [boundary[1] for _, start, end in self.pending for boundary in (start, end)]
        valid = (all(value is not None for value in values)
                 and all(b >= a for a, b in zip(values, values[1:]))
                 and final > initial)
        energy = (final - initial) / 1000 if valid else None
        durations = [end[0] - start[0] for _, start, end in self.pending]
        total_seconds = sum(durations)
        assigned = 0.
        for index, ((key, start, end), seconds) in enumerate(zip(self.pending, durations)):
            joules = None
            if energy is not None:
                joules = (energy - assigned if index == len(self.pending) - 1
                          else energy * seconds / total_seconds)
                assigned += joules
            self.allocations[key] = dict(seconds=seconds, gpu_energy_j=joules)
        self.pending.clear()
        return self.allocations


class Energy:
    """One stage, two cumulative mJ reads; no merging or power fallback."""

    def __init__(self, device, interval=.1):
        # Keep the existing constructor signature for runtime compatibility.
        self.device = device
        self.counter = TotalEnergyCounter(device)
        self.mode = 'nvml_total' if device.type == 'cuda' else 'unavailable'
        self.joules = None
        self.reason = ''

    def __enter__(self):
        try:
            self.start, self.initial = self.counter.read()
            return self
        except BaseException:
            self.counter.close()
            raise

    def __exit__(self, *exc):
        try:
            self.end, final = self.counter.read()
            self.seconds = self.end - self.start
            if self.initial is not None and final is not None:
                self.joules = (final - self.initial) / 1000
                if final == self.initial:
                    self.reason = 'Counter did not advance; interval may be shorter than hardware update period'
        finally:
            self.counter.close()

    def result(self):
        return dict(seconds=self.seconds, gpu_energy_j=self.joules,
                    energy_method=self.mode,
                    energy_reason=self.counter.reason or self.reason)
