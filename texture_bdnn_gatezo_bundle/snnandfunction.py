import torch
def spike_fn(x):
    x = torch.Tensor(x)
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out