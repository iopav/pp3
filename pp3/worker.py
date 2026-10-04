"""Legacy SPSN/TP worker retained for earlier audit reproduction only.

The current controller launches original repository scripts, not this worker.
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .data import build_loaders
from .models import build, batch, export_model, scheduler_step
from .flops import FlopCounter
from .energy import Energy


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_metrics(output,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run_phase(config,model,optimizers,loader,device,train,schedulers=()):
    model.train(train)
    count=FlopCounter()
    count.stage='state_and_bookkeeping'
    total=correct=0
    loss_sum=wait=0.
    with Energy(device,config['power_sample_seconds']) as energy:
        # Account for creating workers/iterator and waiting for CPU data separately.
        tick=time.perf_counter()
        iterator=iter(loader)
        wait+=time.perf_counter()-tick
        with count:
            while True:
                count.paused=True
                tick=time.perf_counter()
                try:
                    x,y=next(iterator)
                except StopIteration:
                    wait+=time.perf_counter()-tick
                    break
                wait+=time.perf_counter()-tick
                x,y=x.to(device),y.to(device)
                count.paused=False
                with torch.set_grad_enabled(train):
                    out,loss=batch(config,model,optimizers,x,y,train,count)
                # Metric computation is not part of training FLOPs.
                count.paused=True
                if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite output or loss')
                total+=len(y)
                correct+=int((out.argmax(1)==y).sum())
                loss_sum+=float(loss.detach())*len(y)
            count.paused=False
            for scheduler in schedulers:
                with count.section('scheduler'):
                    scheduler_step(scheduler,count)
    if not total:
        raise ValueError('No samples processed')
    stats=count.summary()
    stats.update(energy.result(),samples=total,accuracy=correct/total,loss=loss_sum/total,
                 data_wait_seconds=wait)
    return stats


def execute(config):
    output=Path(config['output'])
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',config)
    torch.set_num_threads(config['threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU task requested but CUDA is unavailable')
    train_loader,test_loader,prepare_seconds=build_loaders(config)
    write_json(output/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
               numpy=np.__version__,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
               data_setup_seconds=prepare_seconds,train_samples=len(train_loader.dataset),test_samples=len(test_loader.dataset)))
    counter=FlopCounter()
    counter.stage='prepare'
    with Energy(device,config['power_sample_seconds']) as energy:
        with counter:
            model,opts,schedulers=build(config,device,counter)
    prep=counter.summary(); prep.update(energy.result())
    write_json(output/'prepare_flops.json',prep)
    if not prep['complete']:
        raise RuntimeError('Uncovered preparation operators; see prepare_flops.json')
    rows=[]
    cumulative=prep['flops']
    for epoch in range(1,config['model']['epochs']+1):
        row=dict(method=config['method'],dataset=config['dataset'],seed=config['seed'],epoch=epoch,
                 smoke=config.get('smoke',False))
        for phase,loader,train in [('train',train_loader,True),('test',test_loader,False)]:
            result=run_phase(config,model,opts,loader,device,train,schedulers if train else ())
            write_json(output/f'epoch_{epoch:04d}_{phase}_operations.json',result)
            for key in ['accuracy','loss','samples','seconds','data_wait_seconds','gpu_energy_j','energy_method',
                        'energy_reason','complete','flops','known_flops','fft_flops_estimated']:
                row[f'{phase}_{key}']=result[key]
            if not result['complete']:
                write_json(output/'incomplete_epoch.json',row)
                row['cumulative_flops_including_prepare']=None
                rows.append(row)
                write_metrics(output,rows)
                raise RuntimeError(f'Uncovered {phase} operators; see epoch operation report')
            cumulative+=result['flops']
        row['cumulative_flops_including_prepare']=cumulative
        row['epoch_flops']=row['train_flops']+row['test_flops']
        row['epoch_seconds']=row['train_seconds']+row['test_seconds']
        row['epoch_gpu_energy_j']=(row['train_gpu_energy_j']+row['test_gpu_energy_j']
            if row['train_gpu_energy_j'] is not None and row['test_gpu_energy_j'] is not None else None)
        row['parameter_sha256']=digest(model)
        rows.append(row)
        write_metrics(output,rows)
        print(f'epoch={epoch} train_acc={row["train_accuracy"]:.4f} test_acc={row["test_accuracy"]:.4f} train_flops={row["train_flops"]}',flush=True)
    path=output/'model.pt'
    export_model(model,config,path)
    for row in rows:
        row['final_model_bytes']=path.stat().st_size
    write_metrics(output,rows)
    write_json(output/'status.json',dict(status='success',model_bytes=path.stat().st_size,
               epochs=len(rows),total_flops=cumulative,flops_complete=True))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    try:
        execute(config)
    except Exception:
        output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
        error=traceback.format_exc()
        write_json(output/'status.json',dict(status='failed',traceback=error))
        print(error,flush=True)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
