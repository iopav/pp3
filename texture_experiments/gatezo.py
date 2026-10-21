"""Isolated entry calling the original Texture GateZO implementation."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import traceback

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from data import add_data_arguments, load_data, data_metadata
from energy import AuxiliaryMemory, Energy
from growth_policy import grow_until_neurons

BUNDLE = Path(__file__).resolve().parents[1] / 'texture_bdnn_gatezo_bundle'
sys.path.insert(0, str(BUNDLE))
from gate_zo_bdnn.bdnn1_gatezo import BDNN1GateZO
from gate_zo_bdnn.config_gatezo import GATE_ZO_DEFAULTS
from gate_zo_bdnn.structural_gate import classification_metrics, run_gatezo_growth

target_growth = grow_until_neurons(run_gatezo_growth)


class BoundedGateZO(BDNN1GateZO):
    """Bound actual neurons without changing the bundle or its growth counter."""
    def __init__(self, max_hidden=49, **kwargs):
        super().__init__(**kwargs)
        self.max_hidden = max_hidden

    @property
    def hidden_count(self):
        return 0 if isinstance(self.W, list) else int(self.W.shape[1])

    def sc_Search(self, x, targets):
        remaining = self.max_hidden - self.hidden_count
        if remaining <= 0:
            return [], [], 1
        original = self.nB
        self.nB = min(original, remaining)
        try:
            return super().sc_Search(x, targets)
        finally:
            self.nB = original

    def addNodes(self, w, u):
        if self.hidden_count + w.shape[1] > self.max_hidden:
            raise RuntimeError('GateZO proposal exceeds actual neuron limit')
        super().addNodes(w, u)

    def classification(self, x, targets, x_test, test_targets):
        return target_growth(self, x, targets, x_test, test_targets, self.structural_gate)


def array_bytes(value):
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else np.asarray(value).nbytes


class Recorder:
    # This is an adapter to the bundle's existing hook interface, not a PP3 runtime.
    counter = None

    def __init__(self, model, x, targets, x_test, t_test, output):
        self.model, self.x, self.targets = model, x, targets
        self.x_test, self.t_test, self.output = x_test, t_test, output
        self.outer_array_indices = model.structural_gate.make_outer_split(targets)
        self.rows, self.parts = [], []
        self.total_energy, self.total_seconds = 0.0, 0.0
        self.meter = None
        self.memory = None
        self.epoch = 0

    def begin(self, phase, epoch=None):
        self.phase = phase
        if epoch is not None:
            self.epoch = epoch
        if phase != 'test' and self.meter is None:
            self.meter = Energy(self.model.compute_device)
            self.meter.__enter__()
            parameter_device_bytes = sum(
                array_bytes(value) for value in (self.model.W, self.model.U, self.model.Beta)
                if isinstance(value, torch.Tensor)
                and value.device == self.model.compute_device
            )
            self.memory = AuxiliaryMemory(
                self.model.compute_device, parameter_device_bytes
            )
            self.memory.__enter__()

    def end(self, accuracy=None, loss=None):
        if self.phase != 'test':
            self.finish_measurement()
        if self.phase == 'final_refit_and_test':
            self.epoch_row(self.epoch, unit='final_refit_and_test')

    def finish_measurement(self):
        """Stop the current train-side window before any test evaluation."""
        if self.meter is None:
            return
        self.memory.__exit__(None, None, None)
        self.meter.__exit__(None, None, None)
        self.parts.append({**self.meter.result(), **self.memory.result()})
        self.memory = None
        self.meter = None

    def epoch_row(self, epoch, unit='growth_round'):
        m = self.model
        train = test = dict(accuracy=None, mse=None)
        if m.hidden_count and not isinstance(m.Beta, list):
            train = classification_metrics(m.getH(self.x)[0], self.targets, m.Beta)
            test = classification_metrics(m.getH(self.x_test)[0], self.t_test, m.Beta)
        seconds = sum(p['seconds'] for p in self.parts)
        joules = (sum(p['gpu_energy_j'] for p in self.parts)
                  if all(p['gpu_energy_j'] is not None for p in self.parts) else None)
        auxiliary = (max(p['auxiliary_mem_bytes'] for p in self.parts)
                     if all(p['auxiliary_mem_bytes'] is not None for p in self.parts) else None)
        batch = min(m.compute_batch_size, len(self.x))
        recurrent_state = batch * m.hidden_count * 3 * 4
        hidden_activation = batch * self.x.shape[1] * m.hidden_count * 4
        hidden_state = recurrent_state + hidden_activation
        if auxiliary is not None and auxiliary < hidden_state:
            raise RuntimeError('Measured auxiliary memory is smaller than hidden working storage')
        self.total_seconds += seconds
        self.total_energy = (self.total_energy + joules
                             if self.total_energy is not None and joules is not None else None)
        row = dict(epoch=epoch, unit='epoch' if unit.startswith('growth_round') else unit,
            source_unit=unit, hidden_neurons=m.hidden_count,
            para_mem_bytes=sum(array_bytes(v) for v in (m.W, m.U, m.Beta) if not isinstance(v, list)),
            auxiliary_mem_bytes=auxiliary,
            hidden_state_mem_bytes=hidden_state,
            auxiliary_mem_method=';'.join(sorted(set(p['auxiliary_mem_method'] for p in self.parts))),
            auxiliary_mem_reason=';'.join(sorted(set(
                p['auxiliary_mem_reason'] for p in self.parts if p['auxiliary_mem_reason']))),
            mse=train['mse'], test_mse=test['mse'], train_acc=train['accuracy'], test_acc=test['accuracy'],
            energy_j=joules, power_w=joules / seconds if joules is not None else None,
            total_energy_j=self.total_energy,
            total_power_w=self.total_energy / self.total_seconds if self.total_energy is not None else None,
            epoch_seconds=seconds, total_seconds=self.total_seconds,
            energy_method=';'.join(sorted(set(p['energy_method'] for p in self.parts))),
            energy_reason=';'.join(sorted(set(p['energy_reason'] for p in self.parts if p['energy_reason']))),
            gpu_uuid=self.parts[0]['gpu_uuid'], validation_accuracy=None)
        self.parts = []
        if unit == 'final_refit_and_test':
            (self.output / 'final_metrics.json').write_text(json.dumps(row, indent=2))
        else:
            self.rows.append(row)
            self._csv()
        print(json.dumps(row), flush=True)

    def _csv(self):
        with (self.output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100, help='Compatibility argument; neuron-target policy ignores the loop limit')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--max-hidden', type=int, default=49)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_hidden) < 1:
        parser.error('epochs, batch-size and max-hidden must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    train, test = load_data(args)
    def arrays(dataset):
        indices = range(len(dataset))
        if args.smoke:
            labels = dataset.y if hasattr(dataset, 'y') else np.concatenate([part.y for part in dataset.datasets])
            indices = np.concatenate([np.flatnonzero(labels == label)[:4] for label in np.unique(labels)])
        x, y = zip(*(dataset[i] for i in indices))
        if args.smoke:
            x = [sample[:16] for sample in x]
        return torch.stack(x).numpy(), np.eye(train.classes, dtype=np.float32)[list(y)]
    x, targets = arrays(train)
    x_test, t_test = arrays(test)
    native = dict(GATE_ZO_DEFAULTS)
    native.pop('seed', None)
    native.update(L_max=1 if args.smoke else args.epochs, compute_batch_size=args.batch_size,
                  compute_device=args.device, growth_protocol='validation_checkpoint', gate_zo_seed=args.seed)
    destination = args.output / args.dataset / ('smoke' if args.smoke else 'full') / 'gatezo' / f'seed_{args.seed}'
    destination.mkdir(parents=True, exist_ok=False)
    config = dict(method='gatezo', max_hidden=args.max_hidden, layers=1, native_args=native,
                  source=str(BUNDLE), data_root=str(args.data_root.resolve()), smoke=args.smoke,
                  train_samples=len(x), test_samples=len(x_test), seed=args.seed,
                  epochs_unit='growth_loop', stopping_policy='exact_neuron_target',
                  requested_epochs=args.epochs, epoch_limit_applied=False,
                  torch_version=torch.__version__, deterministic_algorithms=True,
                  cudnn_benchmark=False, cudnn_deterministic=True,
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  measurement_window='growth_search_gate_and_fit_excludes_reporting_evaluations')
    config.update(data_metadata(args, train))
    config['steps'] = x.shape[1]
    (destination / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    recorder = None
    try:
        model = BoundedGateZO(max_hidden=args.max_hidden, **native)
        recorder = Recorder(model, x, targets, x_test, t_test, destination)
        model._pp3_runtime = recorder
        model.classification(x, targets, x_test, t_test)
        if model.hidden_count != args.max_hidden:
            raise AssertionError('Final hidden size differs from target')
        torch.save(dict(W=model.W, U=model.U, Beta=model.Beta, config=config), destination / 'model.pt')
        (destination / 'summary.json').write_text(json.dumps(model.gate_zo_summary, indent=2))
        (destination / 'status.json').write_text(json.dumps(dict(status='success', hidden_neurons=model.hidden_count)))
    except BaseException:
        if recorder is not None and recorder.meter is not None:
            recorder.finish_measurement()
        (destination / 'status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc())))
        raise


if __name__ == '__main__':
    main()
