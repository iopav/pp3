"""Memoize shared FLOP rules for repeated SHD operator signatures, never arithmetic."""
import torch
from pp3.flops import FlopCounter, formula, tensors, argument_metadata


def signature(value):
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), value.dtype)
    if isinstance(value, (list, tuple)):
        return tuple(signature(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, signature(v)) for k, v in sorted(value.items()))
    return (type(value), value)


class RepeatedOpCounter(FlopCounter):
    # These copy/view operations always have zero arithmetic in the shared rule.
    # Their indices affect values but not their FLOPs; tensor shapes still key them.
    COPIES = {'select', 'slice', 'view', 'reshape', 'permute', 'unsqueeze',
              'squeeze', 'expand', 'clone', 'detach', '_to_copy', 'copy_', 'cat', 'stack'}
    FAST = COPIES | {'add', 'sub', 'mul', 'div', 'bmm', 'mm', 'sum', 'gt', 'rsub'}

    def __init__(self, enabled=True):
        super().__init__(enabled)
        self.cached = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.split('::')[-1]
        if name not in self.FAST:
            return super().__torch_dispatch__(func, types, args, kwargs)
        out = func(*args, **kwargs)
        if self.paused or not self.enabled:
            return out
        inputs = signature(tuple(tensors(args))) if name in self.COPIES else signature(args)
        key = (self.stage, func, inputs, signature(kwargs), signature(out))
        record = self.cached.get(key)
        if record is None:
            f, fft, rule = formula(name, args, kwargs, out)
            record = dict(stage=self.stage, operator=str(func),
                inputs=[dict(shape=list(t.shape), dtype=str(t.dtype)) for t in tensors(args)],
                outputs=[dict(shape=list(t.shape), dtype=str(t.dtype)) for t in tensors(out)],
                rule=rule, calls=0, flops_per_call=f, flops=0, fft_flops_estimated=0,
                scalar_arguments=[argument_metadata(v) for v in args if not isinstance(v, torch.Tensor)],
                keyword_arguments=argument_metadata(kwargs))
            self.records[('cached', key)] = record
            self.cached[key] = record
        record['calls'] += 1
        record['flops'] += record['flops_per_call'] or 0
        # FAST excludes FFT operators; those always use the original dispatcher.
        return out
