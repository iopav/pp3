"""Compress the entire source time axis before native label injection."""
import copy
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pp3.data import build_loaders
from pp3.native_runtime import Runtime, MeasuredLoader, write_json


class WindowSumDataset(Dataset):
    def __init__(self, source, source_steps=400, target_steps=10):
        if source_steps != 400 or target_steps != 10:
            raise ValueError('This experiment requires exactly 400 -> 10 time steps')
        self.source = source
        self.source_steps, self.target_steps = source_steps, target_steps
        for name in ('targets', 'origin_indices', 'is_augmented'):
            if hasattr(source, name):
                setattr(self, name, getattr(source, name))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, y = self.source[index]
        if x.ndim != 2 or x.shape[0] != self.source_steps:
            raise ValueError(f'Expected full [400, channels] source, got {tuple(x.shape)}')
        x = x.reshape(self.target_steps, 40, x.shape[1]).sum(dim=1)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError('Non-finite values after time-window summation')
        return x, y


class Spike10Runtime(Runtime):
    def __init__(self):
        super().__init__()
        if self.config['dataset'] != 'spike' or self.config['method'] != 'ffsnn10':
            raise ValueError('Use run_experiments_ffsnn10.py with method ffsnn10 and dataset spike')
        if self.config['data']['steps'] != 400 or self.config['model']['time_steps'] != 10:
            raise ValueError('Expected source steps=400 and model time_steps=10')
        self.transform = dict(source_steps=400, model_steps=10, window_width=40,
            reduction='sum', placement='CPU dataset before label injection',
            scope='train, training-set evaluation, test',
            arithmetic_scope='input preprocessing excluded from model FLOPs',
            full_source_window=True)
        write_json(self.output/'input_transform.json', self.transform)

    def loaders(self, device=False):
        config = copy.deepcopy(self.config)
        # Smoke still limits samples/model size, but must never subsample the
        # 400-step source before the 40-bin sums are formed.
        config['smoke_steps'] = None
        train, test, _ = build_loaders(config)
        result = []
        for source in (train, test):
            dataset = WindowSumDataset(source.dataset)
            loader = DataLoader(dataset, batch_sampler=source.batch_sampler,
                num_workers=config['workers'], generator=source.generator,
                pin_memory=False)
            result.append(MeasuredLoader(loader, self, self.device if device else None))
        return tuple(result)

    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        self.counter.mark_excluded('input_window_sum',
            '400-to-10 CPU window sums are dataset preprocessing; phase wall time includes loader waits')

    def finish(self, state):
        super().finish(state)
        path = self.output/'status.json'
        status = json.loads(path.read_text(encoding='utf-8'))
        status.update(backend='ffsnn_spike10_sum_v1', input_transform=self.transform)
        write_json(path, status)


def from_environment():
    if not os.environ.get('PP3_CONFIG'):
        raise RuntimeError('Launch this experiment through run_experiments_ffsnn10.py')
    return Spike10Runtime()
