"""Offline, CPU-only current-source probes. No dataset downloader imports."""
import argparse
import ast
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['spsn', 'tp', 'syntax'])
parser.add_argument('--dataset', choices=['texture', 'spike'], default='texture')
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(7)
report = dict(method=args.method, dataset=args.dataset, downloaded=False)
try:
    if args.method == 'syntax':
        failures = []
        count = 0
        for folder in ['ESPP-main', 'traces_propagation-main', 'Stochastic-Parallelizable-Spiking-Neuron-SPSN-main']:
            for p in (ROOT/folder).rglob('*.py'):
                count += 1
                try:
                    ast.parse(p.read_text(encoding='utf-8-sig'))
                except SyntaxError as e:
                    failures.append(dict(file=str(p.relative_to(ROOT)), line=e.lineno, error=e.msg))
        report.update(files=count, failures=failures)
    else:
        base = Path(r'D:\workspace\Avalanche\avalanche-master\dataset')/args.dataset
        stem = 'texture' if args.dataset == 'texture' else '20_cla_400steps'
        raw = np.load(base/f'X_train_{stem}.npy', mmap_mode='r')
        labels = np.load(base/f'Y_train_{stem}.npy', mmap_mode='r')
        idx = np.linspace(0,raw.shape[1]-1,16,dtype=int)
        x = torch.tensor(np.array(raw[:4,idx]),dtype=torch.float32)
        y = torch.tensor(np.array(labels[:4]),dtype=torch.long)
        k = 12 if args.dataset=='texture' else 20
        if args.method == 'spsn':
            sys.path.insert(0,str(ROOT/'Stochastic-Parallelizable-Spiking-Neuron-SPSN-main'))
            from network import create_network, train
            net = create_network(dict(neuron='SPSN-SB', nb_layers=1, input_size=x.shape[-1], hidden_size=8, nb_class=k, tau_mem=.02, tau_syn=.02),torch.device('cpu'),16)
            before = [p.detach().clone() for p in net.parameters()]
            result = train(net,[(x,y)],nb_epochs=1)
            report['train_accuracy_on_4_examples'] = float(result['acc'][0])
        else:
            sys.path.insert(0,str(ROOT/'traces_propagation-main'))
            from models.tp_mlp import TP_MLP
            params=dict(l_leak_t=.97,surrogate_type='1',surrogate_scale=1,train_s=False,l_vth=1,l_leak_m=.96,l_rec=False,l_rst_type='soft',l_out_leak_m=1,norm_type='none')
            net=TP_MLP([8],x.shape[-1],k,4,params)
            before=[p.detach().clone() for p in net.parameters()]
            opt=torch.optim.Adam(net.parameters(),lr=.0001)
            net.reset_potential()
            target=torch.nn.functional.one_hot(y,k).float()
            for t in range(16):
                opt.zero_grad()
                out,_=net(x[:,t,:],target)
                net.update()
                opt.step()
                net.detach_membrane_states()
            report['output_shape']=list(out.shape)
        report.update(shape=list(x.shape),weights_changed=any(not torch.equal(a,b) for a,b in zip(before,net.parameters())),finite=all(bool(torch.isfinite(p).all()) for p in net.parameters()),scope='one tiny epoch: 4 training samples, 16 sampled time steps, hidden=8; no accuracy or full-adaptation claim')
    report['status']='completed'
except Exception:
    report.update(status='failed',traceback=traceback.format_exc())
outdir=ROOT/'audit/results_20261003'
outdir.mkdir(exist_ok=True)
(outdir/f'{args.method}_{args.dataset}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
