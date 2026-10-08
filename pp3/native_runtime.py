"""Configuration, data and measurement only; no model or learning algorithm.

Original project entry scripts opt in through PP3_CONFIG. Unknown operations are
retained and make the run's FLOPs status incomplete, without stopping training.
"""
import atexit
import csv
import json
import os
import platform
from pathlib import Path
import random
import sys
import time
import traceback
from contextlib import contextmanager

import numpy as np
import torch
from .data import build_loaders
from .energy import Energy
from .flops import FlopCounter


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   default=str, allow_nan=False), encoding='utf-8')


def check_finite_state(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError('Final model contains non-finite tensors')
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise FloatingPointError('Final model contains non-finite arrays')
    elif isinstance(value, dict):
        for item in value.values():
            check_finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite_state(item)


class MeasuredLoader:
    """Exclude CPU data preparation from model FLOPs, retain elapsed time/energy."""
    def __init__(self, loader, runtime, device=None):
        self.loader, self.runtime, self.device = loader, runtime, device
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        counter = self.runtime.counter
        paused = counter.paused if counter else False
        if counter:
            counter.paused = True
        tick = time.perf_counter()
        iterator = iter(self.loader)
        self.runtime.data_wait += time.perf_counter() - tick
        if counter:
            counter.paused = paused
        while True:
            counter = self.runtime.counter
            paused = counter.paused if counter else False
            if counter:
                counter.paused = True
            tick = time.perf_counter()
            try:
                x, y = next(iterator)
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)
            except StopIteration:
                return
            finally:
                self.runtime.data_wait += time.perf_counter() - tick
                if counter:
                    counter.paused = paused
            yield x, y


class Runtime:
    def __init__(self):
        self.config = json.loads(Path(os.environ['PP3_CONFIG']).read_text(encoding='utf-8'))
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.output = Path(self.config['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.config['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.set_num_threads(self.config['threads'])
        seed = self.config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rows, self.results = [], {}
        self.counter = self.energy = None
        self.data_wait = 0.
        self.complete = True
        self.finished = False
        self.error = None
        write_json(self.output/'config.json', self.config)
        write_json(self.output/'environment.json', dict(python=platform.python_version(),
            torch=torch.__version__, numpy=np.__version__, requested_device=str(self.device),
            entrypoint=sys.argv[0], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
        previous_hook = sys.excepthook
        def hook(kind, value, tb):
            self.error = ''.join(traceback.format_exception(kind, value, tb))
            previous_hook(kind, value, tb)
        sys.excepthook = hook
        atexit.register(self._exit)

    def loaders(self, device=False):
        train, test, seconds = build_loaders(self.config)
        write_json(self.output/'data.json', dict(setup_seconds=seconds,
                   train_samples=len(train.dataset), test_samples=len(test.dataset)))
        destination = self.device if device else None
        return MeasuredLoader(train, self, destination), MeasuredLoader(test, self, destination)

    def arrays(self):
        """Native full-array algorithms keep their existing RAM requirements."""
        tick = time.perf_counter()
        train, test = self.loaders()
        values = []
        for loader in (train, test):
            dataset=loader.dataset
            if loader is train and self.config['method']=='gatezo':
                validation=dataset.validation_dataset
                self.outer_array_indices=(np.arange(len(dataset)),
                    np.arange(len(dataset),len(dataset)+len(validation)))
                dataset=torch.utils.data.ConcatDataset([dataset,validation])
            # Dataset order is deterministic; do not consume shuffled sampler state.
            pairs = [dataset[i] for i in range(len(dataset))]
            values.extend((torch.stack([p[0] for p in pairs]),
                           torch.tensor([p[1] for p in pairs], dtype=torch.long)))
        write_json(self.output/'array_preparation.json', dict(seconds=time.perf_counter()-tick,
                   note='Native full-array input interface; no disk cache'))
        return tuple(values)

    def ordered_loader(self, dataset, device=False):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['model']['batch_size'],
            shuffle=False, num_workers=self.config['workers'],
            generator=torch.Generator().manual_seed(self.config['seed']))
        return MeasuredLoader(loader, self, self.device if device else None)

    @contextmanager
    def preprocessing(self):
        counter = self.counter
        paused = counter.paused if counter else False
        if counter:
            counter.paused = True
        tick = time.perf_counter()
        try:
            yield
        finally:
            self.data_wait += time.perf_counter()-tick
            if counter:
                counter.paused = paused

    def begin(self, phase, epoch=0, unknown=None):
        if self.counter is not None:
            raise RuntimeError('Nested measurement phases are not permitted')
        self.phase, self.epoch, self.data_wait = phase, epoch, 0.
        self.counter = FlopCounter(enabled=self.config.get('count_flops', True))
        self.counter.stage = phase
        if unknown:
            self.counter.mark_unknown('native::uncovered', unknown)
        if not self.config.get('count_flops', True):
            self.counter.mark_unknown('native::disabled', 'Counting disabled for transparency test')
        self.energy = Energy(self.device, self.config['power_sample_seconds'])
        self.energy.__enter__()
        self.counter.__enter__()

    def end(self, accuracy=None, loss=None):
        counter, energy = self.counter, self.energy
        if counter is None:
            raise RuntimeError('No active measurement phase')
        counter.__exit__(None, None, None)
        self.counter = None
        energy.__exit__(None, None, None)
        result = counter.summary()
        result.update(energy.result(), data_wait_seconds=self.data_wait,
                      accuracy=None if accuracy is None else float(accuracy),
                      loss=None if loss is None else float(loss))
        for key in ('accuracy', 'loss'):
            if result[key] is not None and not np.isfinite(result[key]):
                raise FloatingPointError('Non-finite '+key)
        self.complete &= result['complete']
        self.results[(self.epoch, self.phase)] = result
        write_json(self.output/f'epoch_{self.epoch:04d}_{self.phase}_operations.json', result)
        return result

    def epoch_row(self, epoch, unit='epoch'):
        row = dict(method=self.config['method'], dataset=self.config['dataset'],
                   seed=self.config['seed'], epoch=epoch, unit='epoch', source_unit=unit,
                   smoke=self.config.get('smoke', False))
        phases = ['train', 'test'] + [phase for number, phase in self.results
            if number == epoch and phase not in ('train', 'test')]
        for phase in phases:
            result = self.results.get((epoch, phase), {})
            for key in ('accuracy', 'loss', 'seconds', 'data_wait_seconds', 'gpu_energy_j',
                        'energy_method', 'energy_reason', 'complete', 'flops', 'known_flops',
                        'fft_flops_estimated', 'other_flops_estimated'):
                row[f'{phase}_{key}'] = result.get(key)
        measured = [self.results[(epoch, phase)] for phase in phases if (epoch, phase) in self.results]
        row['epoch_flops_complete'] = bool(measured) and all(r['complete'] for r in measured)
        row['epoch_known_flops'] = sum(r['known_flops'] for r in measured)
        row['epoch_flops'] = row['epoch_known_flops'] if row['epoch_flops_complete'] else None
        row['epoch_seconds'] = sum(r['seconds'] for r in measured)
        row['epoch_gpu_energy_j'] = (sum(r['gpu_energy_j'] for r in measured)
            if measured and all(r['gpu_energy_j'] is not None for r in measured) else None)
        self.rows.append(row)
        self._csv()

    def _csv(self):
        if not self.rows:
            return
        fields = list(dict.fromkeys(k for row in self.rows for k in row))
        with (self.output/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self, state):
        check_finite_state(state)
        path = self.output/'model.pt'
        torch.save(state, path)
        for row in self.rows:
            row['final_model_bytes'] = path.stat().st_size
            for (number, phase), result in self.results.items():
                if phase.startswith('final_'):
                    row[phase+'_accuracy'] = result['accuracy']
                    row[phase+'_known_flops'] = result['known_flops']
                    row[phase+'_flops'] = result['flops']
        self._csv()
        self.finished = True
        write_json(self.output/'status.json', dict(
            status='success' if self.complete else 'flops_incomplete',
            training_completed=True, flops_complete=self.complete,
            model_bytes=path.stat().st_size, epochs=len(self.rows), actual_device=str(self.device)))

    def _exit(self):
        if self.counter is not None:
            try:
                self.end()
            except Exception:
                pass
        if not self.finished or self.error:
            write_json(self.output/'status.json', dict(status='failed',
                training_completed=False, reason=self.error or 'Original entry did not finish'))


def from_environment():
    return Runtime() if os.environ.get('PP3_CONFIG') else None
