"""Portable controller. Does not import torch or initialize CUDA in the parent."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
import constants

ROOT=Path(__file__).resolve().parent
# Cluster defaults live in the top-level entry, as requested. CLI paths override these.
DATA_PATHS={'shd':'/data/SHD','spike':'/data/spike','texture':'/data/texture'}


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
    parser.add_argument('--data-root',help='Parent containing spike/, texture/, shd/')
    for name in constants.DATASETS:
        parser.add_argument(f'--{name}-path')
    parser.add_argument('--output',default=constants.RUNTIME['output'])
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--workers',type=int,default=constants.RUNTIME['workers'])
    parser.add_argument('--threads',type=int,default=constants.RUNTIME['threads'])
    parser.add_argument('--smoke',action='store_true',help='1 epoch, tiny data/model; classwise methods retain every class; see configuration')
    parser.add_argument('--smoke-full-time',action='store_true',help='With --smoke: retain all original time steps')
    parser.add_argument('--dry-run',action='store_true',help='Print resolved tasks without creating files or running models')
    args=parser.parse_args()
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
    run_dir=Path(args.output).expanduser().resolve()/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    pending=[]
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
                    c['data']['root']=str(Path(DATA_PATHS[dataset]).expanduser().resolve())
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
                c['output']=str(run_dir/method/dataset/f'seed_{seed}')
                c['entrypoint']=str(entrypoint(c))
                pending.append(c)
    if args.dry_run:
        print(json.dumps(dict(devices=devices,tasks=pending),indent=2)); return 0
    run_dir.mkdir(parents=True,exist_ok=False)
    active={}; finished=[]
    try:
        while pending or active:
            for gpu in devices:
                if pending and gpu not in active:
                    c=pending.pop(0); c['device']='cpu' if gpu=='cpu' else 'cuda:0'; c['gpu_token']=gpu
                    out=Path(c['output']); out.mkdir(parents=True)
                    cfg=out/'launch.json'; cfg.write_text(json.dumps(c,indent=2),encoding='utf-8')
                    env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES='' if gpu=='cpu' else gpu,
                               OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),PYTHONUNBUFFERED='1')
                    log=(out/'run.log').open('w',encoding='utf-8')
                    env['PP3_CONFIG']=str(cfg)
                    env['PYTHONPATH']=os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH')]))
                    env['PYTHONHASHSEED']=str(c['seed'])
                    script=entrypoint(c)
                    try:
                        proc=subprocess.Popen([sys.executable,str(script)],cwd=script.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                    except OSError as error:
                        log.write(str(error)); log.close()
                        (out/'status.json').write_text(json.dumps(dict(status='failed',
                            training_completed=False,reason=str(error)),indent=2),encoding='utf-8')
                        finished.append(dict(method=c['method'],dataset=c['dataset'],seed=c['seed'],
                            device=gpu,exit_code=1,output=c['output']))
                        continue
                    active[gpu]=(proc,log,c)
                    print(f'start {c["method"]}/{c["dataset"]}/{c["seed"]} device={gpu}',flush=True)
            for gpu,(proc,log,c) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    status_path=Path(c['output'])/'status.json'
                    try:
                        status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
                    except (OSError, ValueError) as error:
                        status=dict(status='failed',training_completed=False,reason='Invalid status report: '+str(error))
                    if not status:
                        status=dict(status='failed', training_completed=False,
                                    reason='Entry failed before reporting status; see run.log', process_exit_code=code)
                        status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
                    # A script exit of zero does not certify a complete measurement.
                    if code==0 and status.get('status')!='success':
                        code=2
                    log.close(); finished.append(dict(method=c['method'],dataset=c['dataset'],seed=c['seed'],device=gpu,exit_code=code,output=c['output']))
                    del active[gpu]
                    print(f'finish {c["method"]}/{c["dataset"]}/{c["seed"]} exit={code}',flush=True)
            if active:
                time.sleep(.2)
    except BaseException:
        for proc,log,c in active.values():
            proc.terminate(); proc.wait(); log.close()
        raise
    (run_dir/'tasks.json').write_text(json.dumps(finished,indent=2),encoding='utf-8')
    rows=[]
    for task in finished:
        path=Path(task['output'])/'metrics.csv'
        if path.exists():
            with path.open(encoding='utf-8',newline='') as f:
                for row in csv.DictReader(f):
                    row.update(task_exit_code=task['exit_code'],task_output=task['output'])
                    rows.append(row)
    with (run_dir/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row)) or ['method','dataset','seed','epoch']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Results: {run_dir}',flush=True)
    return int(any(t['exit_code'] for t in finished))


if __name__=='__main__':
    raise SystemExit(main())
