"""Copied native main_exp1, dataset replacement and observation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / 'bdnn_native'
ARTIFACTS = ('model.pt', 'metrics.csv', 'summary.json', 'final_metrics.json',
             'suitW_old_20cla.npy', 'suitU_old_20cla.npy', 'suitB_old_20cla.npy',
             'Ratelist_for_20_old_classes.npy', 'RateList2_for_20_old_classes.npy',
             'error_old_6cla.npy', 'rate_old_6cla.npy')


def reuse(destination, config):
    if not destination.exists():
        return False
    try:
        previous = json.loads((destination/'config.json').read_text(encoding='utf-8'))
        status = json.loads((destination/'status.json').read_text(encoding='utf-8'))
        valid = (previous == config and status.get('status') == 'success'
                 and all((destination/name).is_file() and (destination/name).stat().st_size
                         for name in ARTIFACTS))
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise FileExistsError(f'{destination}: existing run differs or is incomplete; use a new --output')
    print(f'[skip] bdnn: completed native run at {destination}', flush=True)
    return True


def main():
    from data import add_data_arguments, default_data_root
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--device', default='cuda:0', help='Board telemetry only; native BDNN uses CPU')
    for flag in ('epochs', 'batch-size', 'seed'):
        parser.add_argument('--'+flag, type=int, help='Shared-launcher compatibility; NOT applied to native BDNN')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        parser.error('Native BDNN does not truncate data or change parameters for --smoke. Use a separate small dataset fixture.')
    from startup_power import report_startup_power
    report_startup_power(args.device)
    for key in ('epochs', 'batch_size', 'seed'):
        if getattr(args, key) is not None:
            print(f'[bdnn native] --{key.replace("_", "-")} is not applied; preserving original execution', flush=True)
    data_root = (args.data_root or default_data_root(args.dataset)).resolve()
    import runpy
    defaults = runpy.run_path(str(NATIVE/'config.py'))['SCN_DEFAULTS']
    config = dict(method='bdnn', dataset=args.dataset, data_root=str(data_root),
        shd_dt=args.shd_dt, shd_max_time=args.shd_max_time,
        shd_channel_group_size=args.shd_channel_group_size,
        native_args=defaults, random_seed=None, compute_device='cpu', telemetry_device=args.device,
        protocol='native_main_exp1_dataset_only_v1',
        accuracy_reporting='native_per_growth_test_and_test_selected_final',
        energy_measurement='native_phase_boundaries_final_refit_includes_native_final_test',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted([*NATIVE.glob('*.py'), ROOT/'bdnn.py',
                                        ROOT/'native_bdnn_metrics.py', ROOT/'data.py', ROOT/'energy.py'])})
    destination = (args.output/args.dataset/'full'/'bdnn'/'native').resolve()
    if reuse(destination, config):
        return 0
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    env = os.environ.copy()
    for key in ('PP3_CONFIG', 'PP3_CONFIG_JSON', 'PYTHONHASHSEED'):
        env.pop(key, None)
    env['TEXTURE_BDNN_CONFIG'] = str(destination/'config.json')
    try:
        result = subprocess.run([sys.executable, str(NATIVE/'main_exp1.py')], cwd=destination, env=env)
        success = result.returncode == 0 and all((destination/name).is_file() for name in ARTIFACTS)
        (destination/'status.json').write_text(json.dumps(dict(status='success' if success else 'failed',
                                                              returncode=result.returncode)), encoding='utf-8')
        return result.returncode or (0 if success else 1)
    except BaseException as error:
        (destination/'status.json').write_text(json.dumps(dict(status='failed', error=str(error))), encoding='utf-8')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
