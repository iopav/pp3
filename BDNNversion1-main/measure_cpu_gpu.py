#!/usr/bin/env python3
"""Linux hardware-domain power meter (not process-exclusive energy).

python measure_cpu_gpu.py --idle-seconds 15 --output bdnn_power -- python main_exp1.py
Outputs PREFIX.json and PREFIX.csv. Child stdout/stderr remain visible.
"""
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class Sensor:
    name: str
    kind: str                 # power (W) or energy (J)
    scope: str
    source: str
    read: object
    wrap: float = None


def number(path, scale=1.):
    return float(Path(path).read_text().strip()) * scale


def discover_cpu():
    sensors, errors, seen = [], [], set()
    # Grace CPU and module sensors are distinct: module may include GPU and
    # must never be summed with CPU/GPU. Report module only as supplemental.
    for meta in sorted(set(glob.glob('/sys/class/hwmon/hwmon*/device/power*_oem_info') +
                           glob.glob('/sys/class/hwmon/hwmon*/power*_oem_info') +
                           glob.glob('/sys/class/hwmon/hwmon*/power*_label'))):
        try:
            label = Path(meta).read_text().strip()
            if not label.startswith(('Grace Power Socket', 'CPU Power Socket', 'Module Power Socket')):
                continue
            stem = re.sub(r'_(oem_info|label)$', '', meta)
            path = next((Path(stem+s) for s in ('_average', '_input') if Path(stem+s).exists()), None)
            if path is None or str(path.resolve()) in seen:
                continue
            seen.add(str(path.resolve()))
            number(path, 1e-6)
            scope = 'module (may overlap CPU/GPU)' if label.startswith('Module') else 'CPU socket'
            sensors.append(Sensor(label, 'power', scope, str(path), lambda p=path: number(p, 1e-6)))
        except (OSError, ValueError) as exc:
            errors.append(f'{meta}: {exc}')
    # RAPL package only: omit nested core/DRAM domains to avoid double counting.
    if not any(s.scope == 'CPU socket' for s in sensors):
        for name in sorted(glob.glob('/sys/class/powercap/*/name') + glob.glob('/sys/class/powercap/*/*/name')):
            root = Path(name).parent
            try:
                label = Path(name).read_text().strip()
                if not label.startswith('package-') or str(root.resolve()) in seen:
                    continue
                seen.add(str(root.resolve()))
                path = root / 'energy_uj'
                number(path, 1e-6)
                wrap = number(root / 'max_energy_range_uj', 1e-6)
                sensors.append(Sensor('CPU '+label, 'energy', 'CPU socket', str(path),
                                      lambda p=path: number(p, 1e-6), wrap))
            except (OSError, ValueError) as exc:
                errors.append(f'{root}: {exc}')
    return sensors, errors


def discover_gpu(selector):
    """Use the requested physical ID/UUID, never silently choose NVML GPU 0."""
    errors = []
    if selector is None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:
            return [], ['GPU unspecified: supply --gpu UUID (or CUDA_VISIBLE_DEVICES)'], lambda: None
        selector = visible.split(',')[0].strip()
        if selector and selector != '-1' and not selector.startswith('GPU-'):
            # Slurm/CUDA ordinals need not equal NVML physical indices. Resolve
            # CUDA device 0 in a short-lived process before measuring idle.
            try:
                out = subprocess.check_output([sys.executable, '-c',
                    'import torch; print("METER_UUID="+str(torch.cuda.get_device_properties(0).uuid))'],
                    text=True, stderr=subprocess.PIPE, timeout=30)
                selector = next(line.split('=', 1)[1] for line in out.splitlines()
                                if line.startswith('METER_UUID=GPU-'))
            except (OSError, subprocess.SubprocessError, StopIteration) as exc:
                return [], [f'Cannot resolve CUDA GPU UUID: {exc}; supply --gpu GPU-UUID explicitly'], lambda: None
    if not selector or selector == '-1':
        return [], ['No GPU selected'], lambda: None
    try:
        import pynvml as nv
        nv.nvmlInit()
        try:
            handle = (nv.nvmlDeviceGetHandleByUUID(selector) if selector.startswith('GPU-')
                      else nv.nvmlDeviceGetHandleByIndex(int(selector)))
            uuid = nv.nvmlDeviceGetUUID(handle)
            name = nv.nvmlDeviceGetName(handle)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            name = name.decode() if isinstance(name, bytes) else name
            label = f'GPU {name} {uuid}'
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(handle)
                sensor = Sensor(label, 'energy', 'GPU device', 'NVML total_energy_mJ',
                                lambda: nv.nvmlDeviceGetTotalEnergyConsumption(handle)/1000.)
            except Exception:
                nv.nvmlDeviceGetPowerUsage(handle)
                sensor = Sensor(label, 'power', 'GPU device', 'NVML power_mW',
                                lambda: nv.nvmlDeviceGetPowerUsage(handle)/1000.)
            return [sensor], errors, nv.nvmlShutdown
        except Exception:
            nv.nvmlShutdown()
            raise
    except Exception as exc:
        errors.append(f'NVML: {exc}; trying nvidia-smi')
    def query():
        out = subprocess.check_output(['nvidia-smi', '-i', selector,
            '--query-gpu=uuid,name,power.draw', '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        fields = next(csv.reader(out.splitlines()))
        return fields[0].strip(), fields[1].strip(), float(fields[2])
    try:
        uuid, name, _ = query()
        return [Sensor(f'GPU {name} {uuid}', 'power', 'GPU device', 'nvidia-smi power.draw',
                       lambda: query()[2])], errors, lambda: None
    except Exception as exc:
        return [], errors+[f'nvidia-smi: {exc}'], lambda: None


def cpu_topology(cpu):
    root = Path(f'/sys/devices/system/cpu/cpu{cpu}')
    def read(name):
        try: return (root / 'topology' / name).read_text().strip()
        except OSError: return None
    return dict(cpu=cpu, socket=read('physical_package_id'), core=read('core_id'),
                numa_nodes=[p.name for p in root.glob('node[0-9]*')])


def observe_process(root_pid, observed, affinities):
    # Include native BLAS/OpenMP threads and descendants. stat processor is
    # last scheduled CPU, not a proof that every CPU used was sampled.
    pending, seen = [root_pid], set()
    while pending:
        pid = pending.pop()
        if pid in seen: continue
        seen.add(pid)
        for task in Path(f'/proc/{pid}/task').glob('[0-9]*'):
            try:
                fields = (task/'stat').read_text().rsplit(')', 1)[1].split()
                observed.add(int(fields[36]))  # field 39, first suffix field is 3
                affinities.update(os.sched_getaffinity(int(task.name)))
                pending.extend(map(int, (task/'children').read_text().split()))
            except (OSError, ValueError, IndexError):
                pass


def summarize(sensor, rows, interval):
    energy = covered = 0.
    gaps = 0
    for (t0, v0), (t1, v1) in zip(rows, rows[1:]):
        dt = t1-t0
        if v0 is None or v1 is None or dt <= 0 or dt > max(3*interval, 2.):
            gaps += 1
            continue
        if sensor.kind == 'power':
            delta = dt*(v0+v1)/2
        else:
            delta = v1-v0
            if delta < 0 and sensor.wrap:
                delta += sensor.wrap
            if delta < 0:
                gaps += 1
                continue
        energy += delta
        covered += dt
    return dict(energy_j=energy if covered else None,
                average_power_w=energy/covered if covered else None,
                covered_seconds=covered, excluded_intervals=gaps,
                valid_samples=sum(v is not None for _, v in rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--idle-seconds', type=float, default=15.)
    p.add_argument('--interval', type=float, default=.5)
    p.add_argument('--gpu', help='Physical GPU UUID recommended; defaults to first CUDA_VISIBLE_DEVICES token')
    p.add_argument('--output', default='bdnn_power', help='Output prefix')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: p.error('provide -- python YOUR_BDNN_ENTRY.py')
    if args.interval <= 0 or args.idle_seconds < 2*args.interval:
        p.error('interval must be positive; idle-seconds must be >= 2*interval')
    prefix = Path(args.output).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sensors, errors = discover_cpu()
    gpu, gpu_errors, shutdown = discover_gpu(args.gpu)
    sensors += gpu
    errors += gpu_errors
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    try:
        cpu_info = subprocess.check_output(['lscpu'], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        cpu_info = platform.processor()
    report = dict(host=socket.gethostname(), cpu_info=cpu_info, command=command,
        cwd=str(Path.cwd()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        allowed_cpu_topology=[cpu_topology(c) for c in allowed], diagnostics=errors,
        scope='Whole CPU socket / GPU device, not process-exclusive. Baseline is pre-run background load.',
        timing='Entire child command including imports, data loading and saving; sampled endpoints include launch/exit observation overhead.')
    print(cpu_info, flush=True)
    print('Host:', report['host'], 'Allowed CPUs:', allowed, flush=True)
    for s in sensors: print('Sensor:', s.name, '|', s.scope, '|', s.source, flush=True)
    for error in errors: print('Diagnostic:', error, flush=True)
    if not any(s.scope == 'CPU socket' for s in sensors):
        print('CPU power/energy unavailable: no readable CPU socket sensor.', flush=True)
    phases = {}
    observed, affinities = set(), set()
    origin = time.monotonic()
    raw_path = Path(str(prefix)+'.csv')
    process = None
    returncode = 1
    try:
        with raw_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['phase','sensor','time_since_meter_start_s','kind','value_W_or_J','error'])
            def sample(phase, values):
                for i, s in enumerate(sensors):
                    error = ''
                    try:
                        value = float(s.read())
                        if not math.isfinite(value) or value < 0: raise ValueError('invalid reading')
                    except Exception as exc:
                        value, error = None, str(exc)
                    tick = time.monotonic()
                    values[i].append((tick, value))
                    writer.writerow([phase,s.name,tick-origin,s.kind,value,error])
                stream.flush()
            print(f'Measuring pre-run idle baseline for {args.idle_seconds:g} s...', flush=True)
            values = [[] for _ in sensors]
            start = time.monotonic()
            sample('idle', values)
            while time.monotonic()-start < args.idle_seconds:
                time.sleep(min(args.interval, max(0., args.idle_seconds-(time.monotonic()-start))))
                sample('idle', values)
            report['idle_wall_seconds'] = time.monotonic()-start
            phases['idle'] = [summarize(s,r,args.interval) for s,r in zip(sensors,values)]
            values = [[] for _ in sensors]
            sample('run', values)
            start = time.monotonic()
            try:
                process = subprocess.Popen(command)
                report['child_pid'] = process.pid
                while True:
                    observe_process(process.pid, observed, affinities)
                    if process.poll() is not None: break
                    try: process.wait(timeout=args.interval)
                    except subprocess.TimeoutExpired: pass
                    sample('run', values)
                returncode = process.returncode
            except KeyboardInterrupt:
                report['interrupted'] = True
                if process and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                returncode = 130
            except OSError as exc:
                report['launch_error'] = str(exc)
                returncode = 127
            report['command_wall_seconds'] = time.monotonic()-start
            sample('run', values)
            phases['run'] = [summarize(s,r,args.interval) for s,r in zip(sensors,values)]
    finally:
        shutdown()
    report['returncode'] = returncode
    report['observed_cpu_topology'] = [cpu_topology(c) for c in sorted(observed)]
    report['observed_thread_allowed_cpus'] = sorted(affinities)
    report['cpu_observation_note'] = 'Sampled last CPU of child threads/descendants; migrations between samples may be missed.'
    report['sensors'] = []
    print('\nRESULTS (N/A means unavailable; net values are not clipped at zero)', flush=True)
    print('Observed CPU locations:', report['observed_cpu_topology'])
    print(f'Command elapsed: {report["command_wall_seconds"]:.3f} s | Exit: {returncode}')
    print(f'Idle baseline elapsed: {report["idle_wall_seconds"]:.3f} s')
    if not any(s.scope == 'CPU socket' for s in sensors):
        print('CPU idle/run/net W and J: N/A (no readable CPU socket sensor)')
    if not any(s.scope == 'GPU device' for s in sensors):
        print('GPU idle/run/net W and J: N/A (no readable selected GPU sensor)')
    print(f'{"Sensor":48} {"Idle W":>10} {"Run W":>10} {"Net W":>10} {"Run J":>12} {"Net J":>12} {"Covered s":>10}')
    def fmt(value): return 'N/A' if value is None else f'{value:.3f}'
    for i,s in enumerate(sensors):
        idle, run = phases['idle'][i], phases['run'][i]
        baseline, active = idle['average_power_w'], run['average_power_w']
        net = active-baseline if active is not None and baseline is not None else None
        net_j = net*run['covered_seconds'] if net is not None else None
        row = dict(name=s.name,scope=s.scope,source=s.source,method=s.kind,
                   idle=idle,run=run,net_average_power_w=net,net_energy_j=net_j)
        report['sensors'].append(row)
        print(f'{s.name:48} '+ ' '.join(f'{fmt(v):>10}' for v in
              (baseline,active,net,run['energy_j'],net_j,run['covered_seconds'])))
    report['formula'] = 'net_W = run_J / covered_s - idle_W; net_J = run_J - idle_W * covered_s'
    Path(str(prefix)+'.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Saved:', str(prefix)+'.json', raw_path, flush=True)
    return returncode if returncode >= 0 else 128-returncode


if __name__ == '__main__':
    sys.exit(main())
