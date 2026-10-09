"""Portable controller. Does not import torch or initialize CUDA in the parent."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import signal
import traceback
import constants
from importlib.metadata import version, PackageNotFoundError

ROOT=Path(__file__).resolve().parent


def record_exit(config, log, process_code, interruption=None):
    """Persist controller diagnostics after the child has stopped writing."""
    path = Path(config['output'])/'status.json'
    try:
        status = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if not isinstance(status, dict):
            raise ValueError('expected a JSON object')
    except (OSError, ValueError) as error:
        status = dict(status='failed', training_completed=False,
                      reason='Invalid status report: '+str(error))
    if interruption or process_code != 0 or not status:
        reason = interruption or status.get('reason')
        if not reason:
            if process_code < 0 and os.name != 'nt':
                try:
                    name = signal.Signals(-process_code).name
                except ValueError:
                    name = str(-process_code)
                reason = f'Child terminated by signal {name}; external cause unavailable to controller'
            else:
                reason = f'Child exited with code {process_code}; no successful completion report; see preceding output'
        status.update(status='failed', training_completed=False, reason=reason)
    code = process_code or (0 if status.get('status') == 'success' else 2)
    status.update(process_exit_code=process_code, exit_code=code)
    lines = [f'finish {config["method"]}/{config["dataset"]}/{config["seed"]} exit={code}',
             f'  status={status.get("status", "failed")} training_completed={status.get("training_completed", False)}',
             f'  process_exit_code={process_code}']
    if status.get('reason'):
        lines.append(str(status['reason']))
    if status.get('status') == 'flops_incomplete':
        lines.append('  FLOPs coverage check failed; see flops_unknown.json')
    # Seek explicitly: the child may have advanced the shared file descriptor.
    log.seek(0, os.SEEK_END)
    log.write('\n[controller]\n'+'\n'.join(lines)+'\n')
    log.flush()
    path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    return code, lines


def controller_signal(signum, frame):
    raise SystemExit(f'Controller received signal {signal.Signals(signum).name}')


def environment_versions():
    result={}
    for package in ['torch','torchvision','numpy','scipy','snntorch','h5py','timm']:
        try:
            result[package]=version(package)
        except PackageNotFoundError:
            result[package]=None
    return result


def entrypoint(config):
    """Launch real repository entries; never construct a model in the controller."""
    method, dataset = config['method'], config['dataset']
    entries = {
        'spsn': 'Stochastic-Parallelizable-Spiking-Neuron-SPSN-main/run.py',
        'tp': 'traces_propagation-main/main.py',
        'ffsnn': 'SNN_FF-main/FF_SNN/SNNFF_SHD.py',
        'espp': 'ESPP-main/main.py',
        'bdnn': 'BDNNversion1-main/main_exp1.py',
        'efficienttrain': 'EfficientTrain-main/main_buffer.py',
    }
    if method == 'gatezo':
        return ROOT / ('gate_zo_bdnn/train_shd_radlif_residual_proposal.py' if dataset == 'shd'
                       else f'{dataset}_bdnn_gatezo_bundle/run_texture.py')
    return ROOT / entries[method]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=list(constants.METHODS),default=list(constants.METHODS))
    parser.add_argument('--datasets',nargs='+',choices=list(constants.DATASETS),default=list(constants.DATASETS))
    parser.add_argument('--seeds',nargs='+',type=int,default=constants.SEEDS)
    parser.add_argument('--gpus',nargs='+',default=None,help='GPU ordinal/UUID tokens; default inherited CUDA_VISIBLE_DEVICES or 0')
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--data-root',help='Parent containing spike/, texture/, SHD/; defaults to constants.DATASETS roots')
    for name in constants.DATASETS:
        parser.add_argument(f'--{name}-path')
    parser.add_argument('--output',default=constants.RUNTIME['output'])
    parser.add_argument('--split-dir', help='Shared validation index directory; default OUTPUT/_splits')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--workers',type=int,default=constants.RUNTIME['workers'])
    parser.add_argument('--threads',type=int,default=constants.RUNTIME['threads'])
    parser.add_argument('--smoke',action='store_true',help='1 epoch, tiny data/model; classwise methods retain every class; see configuration')
    parser.add_argument('--smoke-full-time',action='store_true',help='With --smoke: retain all original time steps')
    parser.add_argument('--dry-run',action='store_true',help='Print resolved tasks without creating files or running models')
    parser.add_argument('--force', action='store_true', help='Rerun even if identical completed output exists')
    parser.add_argument('--summarize', nargs='+', help='Legacy alias: summarize existing directories only')
    args=parser.parse_args()
    if args.summarize:
        from summarize_experiments import summarize
        print(summarize(args.summarize, Path(args.output)/'summary.csv'))
        return 0
    if args.epochs is not None and args.epochs<1 or args.workers<0 or args.threads<1:
        parser.error('epochs/threads must be positive and workers nonnegative')
    if args.cpu and args.gpus:
        parser.error('--cpu and --gpus are mutually exclusive')
    if args.smoke_full_time and not args.smoke:
        parser.error('--smoke-full-time requires --smoke')
    inherited=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=['cpu'] if args.cpu else args.gpus if args.gpus is not None else (inherited.split(',') if inherited is not None else ['0'])
    if any(not d or d=='-1' for d in devices) or len(set(devices))!=len(devices):
        parser.error('Invalid or repeated GPU tokens; for no GPU use --cpu')
    # Do not escape an explicitly restricted allocation by replacing its device tokens.
    if not args.cpu and inherited is not None and args.gpus and not set(args.gpus)<=set(inherited.split(',')):
        parser.error('--gpus must use tokens present in inherited CUDA_VISIBLE_DEVICES')
    result_root=Path(args.output).expanduser().resolve()
    from pp3.completed import identity, source_digest, data_signature, completed_index, completed_status
    pending=[]; code_hashes={}; installed=environment_versions()
    for method in dict.fromkeys(args.methods):
        for dataset in dict.fromkeys(args.datasets):
            for seed in dict.fromkeys(args.seeds):
                c=constants.make_config(method,dataset)
                c.update(seed=seed,workers=args.workers,threads=args.threads,smoke=args.smoke)
                path=getattr(args,dataset+'_path')
                if path:
                    c['data']['root']=str(Path(path).expanduser().resolve())
                elif args.data_root:
                    c['data']['root']=str((Path(args.data_root)/('SHD' if dataset=='shd' else dataset)).expanduser().resolve())
                else:
                    c['data']['root']=str(Path(c['data']['root']).expanduser().resolve())
                if args.epochs:
                    c['model']['epochs']=args.epochs
                if args.smoke:
                    c['smoke_samples']=constants.SMOKE['samples']
                    c['smoke_steps']=None if args.smoke_full_time else constants.SMOKE['steps']
                    c['model'].update({k:v for k,v in constants.SMOKE.items() if k not in ('samples','steps')})
                    if method=='ffsnn':
                        c['model']['layers']=2
                    if method=='bdnn':
                        c['smoke_samples']=c['data']['classes']
                        c['model'].update(pool_size=16,growth_step=3)
                    if method=='espp':
                        c['smoke_samples']=c['data']['classes']
                    if method=='efficienttrain':
                        c['model'].update(input_size=64, batch_size=2, warmup_epochs=0)
                    if method=='gatezo':
                        c['model'].update(pool_size=4,growth_step=2,zo_min_keep=1,zo_sample_size=4,
                            readout_bins=8,readout_warmup_bins=0,temporal_windows=2)
                        if dataset!='shd':
                            c['smoke_samples']=4*c['data']['classes']
                            c['model'].update(pool_size=16, growth_step=3, zo_sample_size=0)
                c['output']=str(result_root/dataset/method/f'seed_{seed}')
                c['split_dir']=str(Path(args.split_dir).expanduser().resolve() if args.split_dir else result_root/'_splits')
                c['entrypoint']=str(entrypoint(c))
                key=(method,dataset)
                if key not in code_hashes:
                    code_hashes[key]=source_digest(ROOT,entrypoint(c))
                c.update(source_sha256=code_hashes[key], data_signature=data_signature(c),
                         execution_device='cpu' if args.cpu else 'cuda',
                         python_runtime=dict(executable=sys.executable,version=sys.version,packages=installed))
                c['experiment_id']=identity(c)
                pending.append(c)
    if args.dry_run:
        print(json.dumps(dict(devices=devices,tasks=pending),indent=2)); return 0
    result_root.mkdir(parents=True,exist_ok=True)
    active={}; finished=[]
    existing={} if args.force else completed_index(Path(args.output).expanduser().resolve())
    remaining=[]
    for c in pending:
        previous=existing.get(c['experiment_id'])
        status=completed_status(previous) if previous else None
        if status:
            code=0 if status['status']=='success' else 2
            finished.append(dict(method=c['method'],dataset=c['dataset'],seed=c['seed'],
                device='previous',exit_code=code,output=str(previous),skipped=True,
                experiment_id=c['experiment_id']))
            print(f'skip completed {c["method"]}/{c["dataset"]}/{c["seed"]}: {previous}',flush=True)
        else:
            remaining.append(c)
    pending=remaining
    previous_signals = {s: signal.signal(s, controller_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            for gpu in devices:
                if pending and gpu not in active:
                    c=pending.pop(0); c['device']='cpu' if gpu=='cpu' else 'cuda:0'; c['gpu_token']=gpu
                    out=Path(c['output'])
                    if out.exists():
                        # Only replace the exact selected seed directory, after containment checks.
                        resolved=out.resolve()
                        if out.is_symlink() or result_root not in resolved.parents:
                            raise RuntimeError('Refusing to replace an output outside result root or a symlink') 
                        shutil.rmtree(resolved)
                    out.mkdir(parents=True)
                    env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES='' if gpu=='cpu' else gpu,
                               OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),PYTHONUNBUFFERED='1')
                    log=(out/'run.log').open('w',encoding='utf-8')
                    env['PP3_CONFIG']='environment'
                    env['PP3_CONFIG_JSON']=json.dumps(c)
                    env['PYTHONPATH']=os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH')]))
                    env['PYTHONHASHSEED']=str(c['seed'])
                    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                    script=entrypoint(c)
                    try:
                        proc=subprocess.Popen([sys.executable,str(script)],cwd=script.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                    except OSError as error:
                        _, lines = record_exit(c, log, 1, traceback.format_exc())
                        log.close()
                        print('\n'.join(lines), flush=True)
                        finished.append(dict(method=c['method'],dataset=c['dataset'],seed=c['seed'],
                            device=gpu,exit_code=1,output=c['output']))
                        continue
                    active[gpu]=(proc,log,c)
                    print(f'start {c["method"]}/{c["dataset"]}/{c["seed"]} device={gpu}',flush=True)
            for gpu,(proc,log,c) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    code, lines = record_exit(c, log, code)
                    log.close(); finished.append(dict(method=c['method'],dataset=c['dataset'],seed=c['seed'],device=gpu,exit_code=code,output=c['output']))
                    del active[gpu]
                    print('\n'.join(lines), flush=True)
                    if code:
                        print(f'  log: {Path(c["output"])/"run.log"}', flush=True)
            if active:
                time.sleep(.2)
    except BaseException:
        reason = 'Controller interrupted:\n'+traceback.format_exc()
        for proc,log,c in active.values():
            try:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
                record_exit(c, log, code, reason)
            finally:
                log.close()
        raise
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    print(f'Results: {result_root}; run summarize_experiments.py separately for summary.csv',flush=True)
    return int(any(t['exit_code'] for t in finished))


if __name__=='__main__':
    raise SystemExit(main())
