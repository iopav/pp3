"""Counterexamples against actual functions; no production outputs changed."""
import ast
import contextlib
import collections
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pp3.native_runtime import Runtime
from pp3.completed import completed_status
from pp3.reporting import export_summary
from pp3.energy import Energy

def extract(path, name, ns, cls=None):
    nodes = ast.parse((ROOT/path).read_text(encoding='utf-8-sig')).body
    if cls: nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]

def main():
    results = {}
    ns = dict(np=np, torch=torch, time=time,
              update_list=lambda arr, value, width: np.array([[value]]))
    class Metrics:
        confusion_matrix = staticmethod(lambda a,b: np.eye(2))
    ns['metrics'] = Metrics
    classify = extract('BDNNversion1-main/bdnn1.py', 'classification', ns, 'bdnn1')
    class Fake:
        L=0; L_max=3; tol=.0005; verbose=1
        W=np.zeros((2,0)); U=np.zeros((0,0))
        def sc_Search(self,x,e): return np.ones((2,3)),np.eye(3),0
        def addNodes(self,w,u): self.W,self.U,self.L=w,u,3
        def upgradeSCN(self,x,t): return t,1.
        def getLabel(self,x): return np.eye(2)
        def _accuracy(self,t,o): return 1.
        def getH(self,x): return np.zeros((len(x),self.W.shape[1])),None
        def computeBeta(self,h,t): self.Beta=np.zeros((h.shape[1],t.shape[1]))
        def printProperties(self): pass
    with contextlib.redirect_stdout(io.StringIO()):
        result=classify(Fake(),np.ones((2,2)),np.eye(2),np.ones((2,2)),np.eye(2))
    results['bdnn_first_round_best']=dict(grown_hidden=3,saved_hidden=result[4].shape[1])
    assert result[4].shape[1]==0
    class Progress:
        def __init__(self,values,**kw): self.values=values
        def __iter__(self): return iter(self.values)
        def set_postfix(self,**kw): pass
    train=extract('Stochastic-Parallelizable-Spiking-Neuron-SPSN-main/network.py','train',
                  dict(torch=torch,np=np,time=time,tqdm=Progress))
    values={}
    for batch in (1,2,4):
        model=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad(): model.weight.zero_()
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.ones(4,1,2),torch.zeros(4,dtype=torch.long)),batch_size=batch)
        values[str(batch)]=train(model,loader,nb_epochs=1,lr=0)['loss'][0]
    results['spsn_same_logits_loss_by_batch_size']=values
    assert abs(values['1']/values['4']-4)<1e-6
    path=ROOT/'EfficientTrain-main/utils_buffer.py'
    nodes=ast.parse(path.read_text(encoding='utf-8')).body
    ns=dict(torch=torch,deque=collections.deque,defaultdict=collections.defaultdict)
    for name in ('SmoothedValue','MetricLogger'):
        node=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    meter=ns['MetricLogger']()
    meter.update(class_acc=1.); meter.update(class_acc=0.)
    results['et_tail_batch_accuracy']=dict(batch_sizes=[128,1],correct=[128,0],
        reported=meter.class_acc.global_avg,correct_sample_accuracy=128/129)
    assert meter.class_acc.global_avg==.5
    with tempfile.TemporaryDirectory() as folder:
        r=Runtime.__new__(Runtime); r.output=Path(folder)
        r.config=dict(method='tp',dataset='texture',seed=57,count_flops=False,
                      power_sample_seconds=.1,experiment_id='audit-only')
        r.device=torch.device('cpu'); r.counter=r.energy=None
        r.rows,r.results,r.unknown_records=[],{},[]
        r.begin('train',1); torch.ones(10)+torch.ones(10)
        r.end(accuracy=1.5); r.epoch_row(1)
        r.begin('final_refit'); r.end(accuracy=.75); r.finish({'w':torch.ones(2)})
        results['disabled_flops_and_invalid_accuracy']=dict(
            epoch_flops=r.rows[0]['epoch_flops'],train_accuracy=r.rows[0]['train_accuracy'],
            accepted_as_completed=completed_status(folder) is not None,
            tensor_bytes=8,model_file_bytes=(r.output/'model.pt').stat().st_size)
        out=export_summary(folder,tasks=[dict(method='tp',dataset='texture',seed=57,output=folder)])
        with out.open(encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
        final=next(row for row in rows if not row['epoch'])
        results['final_refit_accuracy_export']=dict(actual_accuracy=.75,exported_test_acc=final['test_acc'])
        assert final['test_acc']==''
        path=r.output/'metrics.csv'
        with path.open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream); fields,row=reader.fieldnames,next(reader)
        row.update(train_accuracy='nan',test_accuracy='')
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerow(row)
        results['nan_accuracy_artifact_accepted']=completed_status(folder) is not None
        assert results['nan_accuracy_artifact_accepted']
    class FrozenCounter:
        reason=''
        def __init__(self,device): self.t=0
        def read(self): self.t+=.01; return self.t,1000
        def close(self): pass
    with patch('pp3.energy.TotalEnergyCounter',FrozenCounter):
        with Energy(torch.device('cuda:0')) as energy: pass
    results['unchanged_energy_counter']=energy.result()
    assert results['unchanged_energy_counter']['gpu_energy_j']==0
    print(json.dumps(results,ensure_ascii=False,indent=2))
    Path(__file__).with_name('reproductions.json').write_text(
        json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
