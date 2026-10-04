"""Run in separate processes: --method spsn/tp. Counter transparency and export."""
import argparse
import copy
import tempfile
from pathlib import Path
import torch
from constants import make_config
from pp3.models import build,batch,export_model,load_inference
from pp3.flops import FlopCounter


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--method',required=True); parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(1)
    c=make_config(args.method,'spike'); c['model'].update(hidden_size=8,layers=1,batch_size=3,epochs=1)
    c['smoke']=True
    device=torch.device(args.device)
    torch.manual_seed(123)
    x=torch.rand(3,16,64,device=device); y=torch.tensor([0,1,2],device=device)
    results=[]
    for enabled in [False,True]:
        torch.manual_seed(57)
        model,opts,_=build(c,device)
        counter=FlopCounter()
        import contextlib
        with counter if enabled else contextlib.nullcontext():
            batch(c,model,opts,x,y,True,counter if enabled else None)
            # Test a partial final batch as in the adapter.
            batch(c,model,opts,x[:1],y[:1],True,counter if enabled else None)
        results.append(([p.detach().clone() for p in model.parameters()],torch.get_rng_state(),
                        torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
        if enabled:
            report=counter.summary()
            assert report['complete'],report['unknown']
            assert report['flops']==sum(r['flops'] for r in report['operations'])
            assert report['flops']==sum(report['known_flops_by_stage'].values())
            assert report['known_flops_by_stage']['optimizer']>0
            assert any(r['operator'].startswith('manual::Adam') for r in report['operations'])
            if args.method=='spsn':
                assert report['fft_flops_estimated']>0
                assert report['known_flops_by_stage']['backward']>0
            else:
                assert report['known_flops_by_stage']['forward_input_and_target']>0
                assert report['known_flops_by_stage']['local_loss_and_gradients']>0
    assert all(torch.equal(a,b) for a,b in zip(results[0][0],results[1][0]))
    assert torch.equal(results[0][1],results[1][1])
    if results[0][2] is not None:
        assert torch.equal(results[0][2],results[1][2])
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/'model.pt'; export_model(model,c,path)
        restored,rc=load_inference(path,device)
        with torch.no_grad():
            torch.manual_seed(999); expected=batch(c,model,[],x,y,False)[0]
            torch.manual_seed(999); actual=batch(rc,restored,[],x,y,False)[0]
        assert torch.equal(expected,actual)
    print(args.method,args.device,'counter transparency, tail batch, export round-trip: PASS')


if __name__=='__main__':
    main()
