"""Legacy SPSN/TP adapter for earlier audit reproduction only.

run_experiments.py now launches original repository scripts directly and never
imports this module. New methods must not be added here.
"""
import sys
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]


def build(config,device,counter=None):
    if config['method'] not in ('spsn', 'tp'):
        raise ValueError('Use run_experiments.py and the original method entry script')
    p,d=config['model'],config['data']
    if config['method']=='spsn':
        sys.path.insert(0,str(ROOT/'Stochastic-Parallelizable-Spiking-Neuron-SPSN-main'))
        from neurons.spsn import SPSN
        sizes=[d['inputs']]+[p['hidden_size']]*p['layers']+[d['classes']]
        steps=(config.get('smoke_steps',16) or d['steps']) if config.get('smoke') else d['steps']
        model=torch.nn.Sequential(*[SPSN(a,b,device,p['spike_mode'],steps,
               fire=i<len(sizes)-2,tau_mem=p['tau_mem'],tau_syn=p['tau_syn'],time_step=p['time_step'])
               for i,(a,b) in enumerate(zip(sizes,sizes[1:]))])
        optimizers=[torch.optim.Adamax(model.parameters(),lr=p['lr'],foreach=False)]
        schedulers=[]
    else:
        sys.path.insert(0,str(ROOT/'traces_propagation-main'))
        from models.tp_mlp import TP_MLP
        model=TP_MLP([p['hidden_size']]*p['layers'],d['inputs'],d['classes'],p['batch_size'],p).to(device)
        groups=[ps for _,ps in model.get_layerwise_params()]
        if not p['layerwise']:
            groups=[[param for group in groups for param in group]]
        optimizers=[torch.optim.Adam(ps,lr=p['lr'],foreach=False) for ps in groups]
        schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=p['epochs'],eta_min=p['lr']*.25) for opt in optimizers]
    if counter is not None:
        # nn.Linear.reset_parameters: sqrt(5), gain (4), std (2), bound (2), negation (1).
        linears=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
        # Original TP LIFLayer constructs fc twice; include the discarded initialization.
        discarded=sum(type(m).__name__=='LIFLayer' for m in model.modules()) if config['method']=='tp' else 0
        counter.add_manual('Linear_initialization_scalars',10,len(linears)+discarded)
        counter.add_manual('Linear_bias_bound_scalars',3,sum(m.bias is not None for m in linears))
        if config['method']=='spsn':
            counter.add_manual('SPSN_decay_and_reinitialization_scalars',12,len(model))
        else:
            counter.add_manual('scheduler_eta_min',1,len(schedulers))
    return model,optimizers,schedulers


def scheduler_step(scheduler,counter):
    # Scalar-only get_lr path in CosineAnnealingLR; integer epoch bookkeeping excluded.
    epoch=scheduler.last_epoch+1
    cost=7 if (epoch-1-scheduler.T_max)%(2*scheduler.T_max)==0 else 12
    counter.add_manual('CosineAnnealingLR_scalars',cost,len(scheduler.optimizer.param_groups))
    scheduler.step()


def optimizer_step(opt,counter):
    # Audited non-capturable real-valued single-tensor Adam/Adamax path.
    # Tensor step increments and tensor updates are already counted by dispatch.
    if counter is not None:
        count=sum(p.grad is not None for group in opt.param_groups for p in group['params'])
        adamax=isinstance(opt,torch.optim.Adamax)
        counter.add_manual('Adamax_scalars' if adamax else 'Adam_scalars',5 if adamax else 9,count)
    opt.step()


def batch(config,model,optimizers,x,y,train,counter=None):
    p=config['model']
    section=counter.section if counter is not None else lambda name: nullcontext()
    if config['method']=='spsn':
        with section('forward_neuron'):
            output=model(x)
        with section('loss'):
            mode=p['loss_mode']
            out=output[:,-1] if mode=='last' else output.max(1)[0] if mode=='max' else output.mean(1)
            loss=F.cross_entropy(out,y)
        if train:
            if p['reg_thr']>0:
                with section('loss'):
                    spikes=torch.stack([layer.nb_spike_per_neuron.sum() for layer in model if layer.fire])
                    threshold=p['reg_thr']*sum(layer.hidden_size for layer in model if layer.fire)
                    if counter is not None:
                        counter.add_manual('SPSN_regularization_threshold',1)
                    loss=loss+F.relu(spikes.sum()-threshold)**2
            optimizers[0].zero_grad()
            with section('backward'):
                loss.backward()
            with section('optimizer'):
                optimizer_step(optimizers[0],counter)
    else:
        model.reset_potential()
        target=F.one_hot(y,config['data']['classes']).float()
        for opt in optimizers:
            opt.zero_grad()
        for t in range(x.shape[1]):
            if train:
                with section('forward_input_and_target'):
                    out,teacher=model(x[:,t,:],target)
                    if counter is not None:
                        counter.add_manual('TP_spike_statistics_host_add',2,p['layers'])
                with section('loss'):
                    loss=F.cross_entropy(out,y)
                    # Original loop computes teacher CE, even though not used in update.
                    F.cross_entropy(teacher,y)
                for opt in optimizers:
                    opt.zero_grad()
                with section('local_loss_and_gradients'):
                    model.update()
                with section('optimizer'):
                    for opt in optimizers:
                        optimizer_step(opt,counter)
                model.detach_membrane_states()
            else:
                with section('forward_neuron'):
                    out=model.inference(x[:,t,:])
        if not train:
            with section('loss'):
                out=F.softmax(out,dim=1)
                loss=F.cross_entropy(out,y)  # Preserve original evaluation loss convention.
    return out,loss


def export_model(model,config,path):
    if hasattr(model,'inference_state'):
        torch.save(dict(format_version=2,config=config,inference_state=model.inference_state()),path)
        return
    # Parameters and persistent configuration, not per-batch neuron traces or optimizer.
    weights={n:p.detach().cpu() for n,p in model.named_parameters() if not n.startswith('target_propagator.')}
    torch.save(dict(format_version=1,config=config,parameters=weights),path)


def load_inference(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    config=saved['config']
    model,_,_=build(config,torch.device(device))
    if 'inference_state' in saved:
        model.restore_inference(saved['inference_state'],device)
        model.eval()
        return model,config
    params={n:p for n,p in model.named_parameters() if not n.startswith('target_propagator.')}
    if set(params)!=set(saved['parameters']):
        raise ValueError('Inference parameter names mismatch')
    with torch.no_grad():
        for name,value in saved['parameters'].items():
            params[name].copy_(value.to(device))
    model.eval()
    return model,config
