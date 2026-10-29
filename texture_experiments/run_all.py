"""Sequential isolated experiment launcher; imports no torch before GPU selection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

METHODS = ('rnn', 'rsnn', 'snnsurrogate', 'ann', 'gatezo', 'bdnn')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', help='Physical GPU ordinal or UUID; default inherited allocation or 0')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--experiments', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--eval-each-epoch', action='store_true',
                        help='Legacy flag: single-epoch energy windows without extra train evaluation')
    args, extra = parser.parse_known_args()
    if args.cpu and args.gpu is not None:
        parser.error('--cpu and --gpu are mutually exclusive')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu = args.gpu if args.gpu is not None else (inherited.split(',')[0] if inherited else '0')
    if not args.cpu and (not gpu or gpu == '-1' or ',' in gpu):
        parser.error('--gpu must select exactly one GPU')
    if not args.cpu and inherited is not None and gpu not in inherited.split(','):
        parser.error('--gpu must be present in inherited CUDA_VISIBLE_DEVICES')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '' if args.cpu else gpu
    env['PYTHONHASHSEED'] = str(args.seed)
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not args.dry_run:
        env.pop('TEXTURE_STARTUP_POWER_REPORTED', None)
        subprocess.run([sys.executable, str(Path(__file__).with_name('startup_power.py')),
                        '--device', 'cpu' if args.cpu else 'cuda:0'], env=env, check=True)
        env['TEXTURE_STARTUP_POWER_REPORTED'] = '1'
    for method in dict.fromkeys(args.experiments):
        command = [sys.executable, str(Path(__file__).with_name(method + '.py')),
                   '--device', 'cpu' if args.cpu else 'cuda:0',
                   '--seed', str(args.seed), *extra]
        if args.eval_each_epoch and method not in ('gatezo', 'bdnn'):
            command.append('--eval-each-epoch')
        print(f'CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]} {command}', flush=True)
        if not args.dry_run:
            result = subprocess.run(command, env=env)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
