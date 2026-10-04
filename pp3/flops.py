"""Leaf-dispatch FLOP accounting inspired by cil_experiments/flops.py.

No Avalanche dependency. Unknown operations fail coverage, never silently count zero.
Counts are conventional arithmetic equivalents, not GPU instruction counts.
"""
import math
from contextlib import contextmanager
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def tensors(obj):
    return [v for v in tree_flatten(obj)[0] if isinstance(v,torch.Tensor)]


def argument_metadata(value):
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,(list,tuple)):
        return [argument_metadata(v) for v in value]
    if isinstance(value,dict):
        return {str(k):argument_metadata(v) for k,v in value.items()}
    return str(value)


ZERO = set('alias as_strided cat clone contiguous copy_ detach empty empty_like empty_strided expand '
           'fill_ flatten flip gather index index_put index_put_ isfinite item lift_fresh masked_fill '
           'masked_fill_ new_empty new_empty_strided new_ones new_zeros ones ones_like permute repeat reshape '
           'select slice slice_backward split split_with_sizes squeeze stack t to transpose unbind unsqueeze '
           'view view_as_real view_as_complex zero_ zeros zeros_like _local_scalar_dense _to_copy arange '
           'scalar_tensor full full_like rand rand_like randn random_ randperm bernoulli bernoulli_ uniform_ '
           'normal_ eq ne equal lt le gt ge argmax argmin sort topk unique _unique2 any all nonzero '
           'threshold threshold_backward relu relu_ hardtanh clamp clamp_ clamp_min clamp_min_ clamp_max '
           'clamp_max_ max min maximum minimum amax amin abs sign sgn where set_ diagonal eye '
           '_unsafe_view unflatten _conj conj resolve_conj is_same_size _reshape_alias scatter scatter_ '
           'constant_pad_nd narrow select_backward _efficientzerotensor detach_'.split())
SPECIAL=set('exp exp_ log log_ sqrt sqrt_ rsqrt reciprocal sigmoid tanh pow pow_'.split())


def formula(name,args,kwargs,out):
    ts=tensors(args)
    os=tensors(out)
    a=ts[0] if ts else None
    n=sum(t.numel() for t in os)
    floating=any(t.is_floating_point() or t.is_complex() for t in ts+os)
    if name.startswith('_record_function_'):
        return 0,0,'profiler_metadata'
    if name.startswith('_foreach_'):
        operation = name[len('_foreach_'):].rstrip('_')
        # In-place foreach operators return (); count the destination list.
        destinations = tensors(args[0]) if name.endswith('_') else os
        elements = sum(t.numel() for t in destinations)
        if not floating or operation in {'abs','maximum','minimum','copy','zero','clamp_min','clamp_max'}:
            if any(t.is_complex() for t in ts):
                return None,0,'complex_foreach_not_audited'
            return 0,0,'foreach_copy_comparison_or_integer'
        if any(t.is_complex() for t in ts+os):
            return None,0,'complex_foreach_not_audited'
        if operation in {'add','sub'}:
            alpha = kwargs.get('alpha', args[2] if len(args)>2 else 1)
            return elements*(1+(alpha != 1)),0,'foreach_elementwise_add_scale'
        if operation in {'mul','div','sqrt','reciprocal','neg','exp','pow'}:
            return elements,0,'foreach_elementwise_arithmetic'
        if operation == 'lerp':
            return 3*elements,0,'foreach_lerp_sub_mul_add'
        if operation in {'addcmul','addcdiv'}:
            value = kwargs.get('value', args[3] if len(args)>3 else 1)
            if isinstance(value, torch.Tensor):
                return None,0,'foreach_tensor_scale_values_not_audited'
            scales = value if isinstance(value, (tuple,list)) else [value]*len(destinations)
            return sum(t.numel()*(2+(v != 1)) for t,v in zip(destinations, scales)),0,'foreach_fused_scaled_arithmetic'
        return None,0,'unknown_foreach'
    if name in {'silu', 'silu_backward'}:
        return n*(2 if name == 'silu' else 6),0,'silu_sigmoid_equivalent_and_product_rule'
    if name in {'index_fill_', 'index_fill', 'one_hot', 'diag_embed', '_linalg_check_errors',
                'median', 'max_pool2d_with_indices'}:
        return 0,0,'copy_integer_or_comparison_only'
    if name == 'resolve_neg':
        return n*(2 if a.is_complex() else 1) if a.is_neg() else 0,0,'materialized_negation_if_needed'
    if name in {'scatter_add', 'scatter_add_'}:
        return args[3].numel()*(2 if args[3].is_complex() else 1),0,'scatter_accumulation'
    if name in {'index_add', 'index_add_'}:
        cost = 2 if args[3].is_complex() else 1
        alpha = kwargs.get('alpha', args[4] if len(args)>4 else 1)
        if alpha != 1:
            if isinstance(alpha, complex):
                return None,0,'complex_index_add_scale_not_audited'
            cost += 2 if args[3].is_complex() else 1
        return args[3].numel()*cost,0,'indexed_accumulation'
    if name == 'matmul':
        return (8 if a.is_complex() else 2)*n*a.shape[-1],0,'broadcast_dense_matmul_mac_2'
    if name == 'std' and not a.is_complex():
        return 4*a.numel()+n,0,'two_pass_mean_center_square_reduce_divide_sqrt_equivalent'
    if name == 'convolution' and not args[6] and not a.is_complex():
        products = args[1].shape[1]*math.prod(args[1].shape[2:])
        return 2*n*products+(n if args[2] is not None else 0),0,'dense_convolution_padded_mac_estimate'
    if name == 'convolution_backward' and not args[7] and not a.is_complex():
        grad, inputs, weights = args[:3]
        mask = args[10]
        products = weights.shape[1]*math.prod(weights.shape[2:])
        f = 2*grad.numel()*products*(int(mask[0])+int(mask[1]))
        if mask[2]:
            f += grad.numel()-grad.shape[1]
        return f,0,'dense_convolution_backward_requested_gradients_estimate'
    if name == 'max_pool2d_with_indices_backward':
        return a.numel(),0,'pool_gradient_scatter_accumulation'
    if name=='linalg_vector_norm' and not a.is_complex():
        order=args[1] if len(args)>1 else 2
        if order==2:
            return 2*a.numel(),0,'norm2_square_reduce_sqrt'
        if order==1:
            return a.numel()-n,0,'norm1_reduce'
    if name in {'softplus','softplus_backward'}:
        return (5 if name=='softplus' else 6)*n,0,'softplus_full_branch_equivalent'
    if name=='multinomial' and args[1]==1:
        return 3*a.numel(),0,'sampling_exponential_race_estimate'
    if name=='uniform_':
        return 2*n,0,'uniform_affine_scale_and_shift_rng_engine_excluded'
    if name in {'normal_','randn'}:
        return None,0,'normal_rng_transform_not_audited'
    if name=='abs' and a.is_complex():
        return 4*n,0,'complex_magnitude_mul_mul_add_sqrt'
    if name in {'scatter','scatter_'} and (kwargs.get('reduce') is not None or len(args)>4):
        return None,0,'unsupported_scatter_reduction'
    if name in {'index_put','index_put_'} and (kwargs.get('accumulate',False) or len(args)>3 and args[3]):
        return args[2].numel()*(2 if args[2].is_complex() else 1),0,'index_accumulate'
    if name in ZERO or name.startswith(('logical_','bitwise_')):
        return 0,0,'non_flop'
    if not floating and name in {'add','sub','mul','div','sum','prod','add_','sub_','mul_','div_'}:
        return 0,0,'integer_arithmetic'
    complex_out=any(t.is_complex() for t in os)
    if name in {'add','add_','sub','sub_','rsub'}:
        cost=2 if complex_out else 1
        if kwargs.get('alpha',1)!=1:
            cost+=(6 if isinstance(kwargs['alpha'],complex) else 2) if complex_out else 1
        return cost*n,0,'elementwise_add_scaled'
    if name in {'mul','mul_','div','div_'}:
        # Complex*complex=4 real multiplies+2 adds; complex/complex=11 equivalents.
        both=(len(ts)>1 and ts[0].is_complex() and ts[1].is_complex()) or any(isinstance(v,complex) for v in args)
        if name.startswith('div') and complex_out and len(ts)>1 and not ts[0].is_complex() and ts[1].is_complex():
            return None,0,'unsupported_real_div_complex'
        factor=(6 if name.startswith('mul') else 11) if both else 2 if complex_out else 1
        return factor*n,0,'elementwise_real_or_complex'
    if name in SPECIAL:
        if complex_out:
            return None,0,'unsupported_complex_special'
        return n,0,'special_function_equivalent_1'
    if name in {'neg','neg_'}:
        return n*(2 if complex_out else 1),0,'negation'
    if name in {'sigmoid_backward','tanh_backward'}:
        return 3*n,0,'activation_backward'
    if name in {'addcmul','addcmul_','addcdiv','addcdiv_'}:
        if complex_out:
            return None,0,'unsupported_complex_fused'
        return n*(2+(kwargs.get('value',1)!=1)),0,'fused_elementwise'
    if name in {'lerp','lerp_'}:
        return 3*n,0,'lerp_sub_mul_add'
    if name in {'mm','bmm','addmm','baddbmm'}:
        offset=1 if name in {'addmm','baddbmm'} else 0
        left,right=args[offset:offset+2]
        k=left.shape[-1]
        factor=8 if left.is_complex() else 2
        f=factor*n*k
        if offset:
            add_cost=2 if left.is_complex() else 1
            if kwargs.get('alpha',1)!=1:
                f+=n*((6 if isinstance(kwargs['alpha'],complex) else 2) if left.is_complex() else 1)
            if kwargs.get('beta',1)!=0:
                f+=n*add_cost
                if kwargs.get('beta',1)!=1:
                    f+=n*((6 if isinstance(kwargs['beta'],complex) else 2) if left.is_complex() else 1)
        return f,0,'dense_matmul_mac_2'
    if name in {'dot','vdot'}:
        return a.numel()*(8 if a.is_complex() else 2),0,'dot'
    if name in {'sum','mean','prod'}:
        f=max(0,a.numel()-n)*((6 if name=='prod' else 2) if a.is_complex() else 1)
        if name=='mean':
            f+=n*(2 if a.is_complex() else 1)
        return f,0,'reduction'
    if name in {'_softmax','_log_softmax'}:
        return (4*a.numel() if name=='_log_softmax' else 4*a.numel()-n//a.shape[args[1]]),0,'softmax_equivalent'
    if name in {'_softmax_backward_data','_log_softmax_backward_data'}:
        return 4*n-n//a.shape[args[2]],0,'softmax_backward_equivalent'
    if name=='nll_loss_forward':
        targets=args[1].numel()
        reduction=args[3]
        # One negation per target, plus reductions; weighted variant includes multiply.
        f=targets*(2 if args[2] is not None else 1)
        if reduction:
            f+=max(0,targets-1)+(reduction==1)
        return f,0,'nll_loss'
    if name=='nll_loss_backward':
        targets=args[2].numel()
        return targets*(1+(args[4]==1)+(args[3] is not None)),0,'nll_backward'
    if name in {'_fft_r2c','_fft_c2r','_fft_c2c'}:
        dims=list(args[1])
        # c2r input stores only half of the last dimension; output has full length.
        full=os[0] if name=='_fft_c2r' else a
        lengths=[full.shape[d] for d in dims]
        volume=math.prod(lengths)
        batch=full.numel()//volume
        coefficient=5. if name=='_fft_c2c' else 2.5
        f=coefficient*batch*volume*sum(math.log2(v) for v in lengths)
        if args[2]!=0: # ortho or inverse-length normalization
            f+=sum(t.numel()*(2 if t.is_complex() else 1) for t in os)
        f=int(math.ceil(f))
        return f,f,'fft_length_formula_estimate'
    return None,0,'unknown'


class FlopCounter(TorchDispatchMode):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled=enabled
        self.records={}
        self.paused=False
        self.stage='other'

    @contextmanager
    def section(self,name):
        previous=self.stage
        self.stage=name
        try:
            yield
        finally:
            self.stage=previous

    def add_manual(self,name,flops,calls=1,rule='host_scalar_arithmetic'):
        """Only arithmetic absent from ATen dispatch, never tensor arithmetic."""
        if not calls:
            return
        key=('manual',self.stage,name,flops,rule)
        if key not in self.records:
            self.records[key]=dict(stage=self.stage,operator='manual::'+name,inputs=[],outputs=[],rule=rule,
                calls=0,flops_per_call=flops,flops=0,fft_flops_estimated=0)
        self.records[key]['calls']+=calls
        self.records[key]['flops']+=flops*calls

    def mark_unknown(self,name,reason,inputs=()):
        key=('unknown',self.stage,name,reason,str(inputs))
        if key not in self.records:
            self.records[key]=dict(stage=self.stage,operator=name,inputs=inputs,outputs=[],rule=reason,
                calls=0,flops_per_call=None,flops=0,fft_flops_estimated=0)
        self.records[key]['calls']+=1

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        out=func(*args,**kwargs)
        if self.paused or not self.enabled:
            return out
        name=func._schema.name.split('::')[-1]
        f,fft,rule=formula(name,args,kwargs,out)
        ins=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors(args)]
        outs=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors(out)]
        # Include formula/count to distinguish alpha/beta/reduction options at same shape.
        key=(self.stage,str(func),str(ins),str(outs),rule,f,fft)
        if key not in self.records:
            self.records[key]=dict(stage=self.stage,operator=str(func),inputs=ins,outputs=outs,rule=rule,
                scalar_arguments=[argument_metadata(v) for v in args if not isinstance(v,torch.Tensor)],
                keyword_arguments=argument_metadata(kwargs),
                calls=0,flops_per_call=f,flops=0,fft_flops_estimated=0)
        r=self.records[key]
        r['calls']+=1
        r['flops']+=f or 0
        r['fft_flops_estimated']+=fft
        return out

    def summary(self):
        rows=list(self.records.values())
        unknown=[r for r in rows if r['flops_per_call'] is None]
        known=sum(r['flops'] for r in rows)
        stages={}
        for r in rows:
            stages[r['stage']]=stages.get(r['stage'],0)+r['flops']
        return dict(complete=not unknown,flops=known if not unknown else None,
            known_flops_by_stage=stages,
            known_flops=known,fft_flops_estimated=sum(r['fft_flops_estimated'] for r in rows),
            other_flops_estimated=sum(r['flops'] for r in rows if 'estimate' in r['rule'] and 'fft_' not in r['rule']),
            unknown=unknown,operations=rows,
            convention='Arithmetic equivalents; FFT formula estimates; special scalar functions=1; MAC=2')
