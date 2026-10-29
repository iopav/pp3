#!/usr/bin/env python3
"""Monitor Grace CPU / Hopper GPU power while executing an experiment.
Usage: python measure_cpu_gpu.py python main_exp1-2.py
"""
import glob
import math
import os
import subprocess
import sys
import threading
import time

INTERVAL = 0.5
PRINT_EVERY = 5.0


def find_grace_sensors():
    found = {}
    for meta in glob.glob('/sys/class/hwmon/hwmon*/device/power1_oem_info') + glob.glob('/sys/class/hwmon/hwmon*/power1_oem_info'):
        root = os.path.dirname(meta)
        reading = os.path.join(root, 'power1_average')
        try:
            with open(meta, encoding='utf-8') as f:
                label = f.read().strip()
            if not os.path.isfile(reading):
                continue
            if label.startswith(('Grace Power Socket', 'CPU Power Socket', 'Module Power Socket')):
                found[label] = reading
        except OSError:
            continue
    return found


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python measure_cpu_gpu.py python main_exp1-2.py')
    try:
        import pynvml as nv
        nv.nvmlInit()
        gpu = nv.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        sys.exit(f'NVML unavailable: {exc}')

    sensors = find_grace_sensors()
    print('Discovered Grace sensors:', sensors or 'none (CPU power unavailable)', flush=True)
    readings = {'GPU board': []}
    for label in sensors:
        readings[label] = []
    stop = threading.Event()
    start = time.perf_counter()

    def sample():
        last_print = start
        while not stop.is_set():
            t = time.perf_counter() - start
            try:
                p = nv.nvmlDeviceGetPowerUsage(gpu) / 1000.0
                if math.isfinite(p) and p >= 0:
                    readings['GPU board'].append((t, p))
            except Exception:
                pass
            for label, path in sensors.items():
                try:
                    with open(path) as f:
                        p = float(f.read().strip()) / 1e6
                    if math.isfinite(p) and p >= 0:
                        readings[label].append((time.perf_counter() - start, p))
                except (OSError, ValueError):
                    pass
            if time.perf_counter() - last_print >= PRINT_EVERY:
                print('[Power] ' + ' | '.join(
                    f'{label}: {values[-1][1]:.2f} W'
                    for label, values in readings.items() if values
                ), flush=True)
                last_print = time.perf_counter()
            stop.wait(INTERVAL)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        process = subprocess.run(sys.argv[1:])
    finally:
        stop.set()
        thread.join()
        duration = time.perf_counter() - start
        print(f'\nMonitored elapsed time: {duration:.3f} s')
        for label, values in readings.items():
            if not values:
                print(f'{label}: unavailable')
                continue
            # Piecewise trapezoidal integration; no extrapolation of uncovered endpoints.
            energy = sum((t1-t0)*(p0+p1)/2 for (t0,p0),(t1,p1) in zip(values,values[1:]))
            covered = values[-1][0]-values[0][0]
            avg = energy/covered if covered>0 else float('nan')
            print(f'{label}: avg={avg:.2f} W, energy={energy:.2f} J, covered={covered:.2f} s')
        nv.nvmlShutdown()
    sys.exit(process.returncode)

if __name__ == '__main__':
    main()
