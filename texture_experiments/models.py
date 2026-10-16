"""Single hidden layer sequence classifiers; no temporal resampling."""
import torch
from torch import nn


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad / (1 + ctx.slope * x.abs()).square(), None


class Classifier(nn.Module):
    def __init__(self, method, config, inputs=9, classes=12):
        super().__init__()
        self.method, self.config = method, config
        h = config['hidden_size']
        self.hidden = (nn.RNN(inputs, h, num_layers=1, batch_first=True,
                              nonlinearity=config['rnn_nonlinearity'])
                       if method == 'rnn' else nn.Linear(inputs, h))
        if method == 'rsnn':
            self.recurrent = nn.Linear(h, h, bias=False)
        self.readout = nn.Linear(h, classes)
        self.recurrent_state_bytes = 0
        self.hidden_activation_bytes = 0

    def forward(self, x):
        h = self.config['hidden_size']
        # Separate recurrent state from the primary B*T*H hidden activation.
        # The latter is present for ANN as well and is needed by backpropagation.
        states = {'ann': 0, 'rnn': 1, 'rsnn': 2, 'snnsurrogate': 1}[self.method]
        self.recurrent_state_bytes = x.size(0) * h * x.element_size() * states
        self.hidden_activation_bytes = (
            x.size(0) * x.size(1) * h * x.element_size()
        )
        if self.method == 'rnn':
            features, _ = self.hidden(x)
            pooled = features.mean(dim=1)
        elif self.method == 'ann':
            pooled = torch.relu(self.hidden(x)).mean(dim=1)
        else:
            mem = x.new_zeros((x.size(0), h))
            previous = torch.zeros_like(mem) if self.method == 'rsnn' else None
            pooled = torch.zeros_like(mem)
            currents = self.hidden(x)
            for current in currents.unbind(dim=1):
                if self.method == 'rsnn':
                    current = current + self.recurrent(previous)
                mem = self.config['spike_beta'] * mem + current
                spike = Spike.apply(mem - self.config['spike_threshold'],
                                    self.config['surrogate_slope'])
                mem = mem - spike.detach() * self.config['spike_threshold']
                if self.method == 'rsnn':
                    previous = spike
                pooled = pooled + spike
            pooled = pooled / x.size(1)
        return self.readout(pooled)
