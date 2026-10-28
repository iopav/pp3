"""Isolated entry calling the original Texture GateZO implementation."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import traceback
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from data import add_data_arguments, load_data, data_metadata
from completion import skip_completed
from energy import TotalEnergyCounter
from train import allocate_block_energy
from metrics import compact_metrics
from bp_memory import BP_MEMORY_VERSION, BP_MEMORY_FIELD
from state_memory import AUX_MEMORY_VERSION, SynchronizedSegment
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


def native_settings(*, rounds=49, growth=1, batch_size=128, device='cuda:0', seed=57):
    """Use the native nB and L_max controls, independent of baseline epochs."""
    if rounds < 1 or growth < 1:
        raise ValueError('L_max and nB must be positive')
    native = dict(GATE_ZO_DEFAULTS)
    native.pop('seed', None)
    native.update(T_max=20, L_max=rounds, nB=growth, compute_batch_size=batch_size,
                  compute_device=device, growth_protocol='validation_checkpoint',
                  gate_zo_seed=seed, gate_zo_verbose=False)
    return native


class Recorder:
    """Five growth-round counter windows; final refit is measured separately."""
    counter = None  # Native hook: no FLOPs instrumentation.

    def __init__(self, model, x, targets, x_test, t_test, output):
        self.model, self.x, self.targets = model, x, targets
        self.x_test, self.t_test, self.output = x_test, t_test, output
        self.outer_array_indices = model.structural_gate.make_outer_split(targets)
        self.rows, self.pending = [], []
        self.total_energy, self.total_seconds = 0.0, 0.0
        self.energy_counter = TotalEnergyCounter(model.compute_device)
        self.initial_energy = None
        self.window_open = False
        self.memory = None
        self.epoch = 0
        self.final_evaluation = None
        self.final_joules = None
        self.train_accuracy = None

    def begin(self, phase, epoch=None):
        if phase == 'test':
            raise RuntimeError('Per-round test reporting is disabled')
        if self.memory is not None:
            raise RuntimeError('Nested training measurement')
        if phase == 'final_refit_and_test':
            self.flush()
        self.phase = phase
        if epoch is not None:
            self.epoch = epoch
        if not self.window_open:
            _, self.initial_energy = self.energy_counter.read()
            self.window_open = True
        if self.model.compute_device.type == 'cuda':
            torch.cuda.synchronize(self.model.compute_device)
        self.started = time.perf_counter()
        self.memory = SynchronizedSegment(self.model.compute_device)
        self.memory.__enter__()

    def end(self, accuracy=None, loss=None):
        self.train_accuracy = accuracy
        self.finish_measurement()
        if self.phase == 'final_refit_and_test':
            self.epoch_row(self.epoch, unit='final_refit_and_test')

    def close_window(self):
        _, final = self.energy_counter.read()
        self.window_open = False
        return ((final - self.initial_energy) / 1000
                if final is not None and self.initial_energy is not None else None)

    def finish_measurement(self):
        """Stop round timing; only the final refit also closes its energy window."""
        if self.memory is None:
            return
        self.memory.__exit__(None, None, None)
        self.seconds = time.perf_counter() - self.started
        self.memory = None
        if self.phase == 'final_refit_and_test':
            # Called by the refit adapter before final train/test evaluation.
            self.final_joules = self.close_window()

    def epoch_row(self, epoch, unit='growth_round'):
        m = self.model
        row = compact_metrics(dict(epoch=epoch, hidden_neurons=m.hidden_count,
            para_mem_bytes=sum(array_bytes(v) for v in (m.W, m.U, m.Beta) if not isinstance(v, list)),
            optimizer_state_bytes=0, auxiliary_mem_bytes=0, **{BP_MEMORY_FIELD: 0},
            energy_j=None, power_w=None, total_energy_j=None, total_power_w=None,
            epoch_seconds=self.seconds, total_seconds=None, validation_accuracy=None,
            train_acc=self.train_accuracy, final_test_acc=None))
        if unit == 'final_refit_and_test':
            self.total_energy, self.total_seconds = allocate_block_energy(
                [row], self.final_joules, self.total_energy, self.total_seconds)
            train, test = self.final_evaluation
            row.update(train_acc=train['accuracy'], final_test_acc=test['accuracy'])
            (self.output / 'final_metrics.json').write_text(json.dumps(row, indent=2))
            self.rows.append(row)
            self.write_csv()
            print(json.dumps(row), flush=True)
        else:
            self.rows.append(row)
            self.pending.append(row)
            if unit == 'growth_round_failed':
                self._csv()

    def flush(self):
        if not self.pending:
            return
        if self.memory is not None:
            raise RuntimeError('Cannot flush an unfinished round')
        joules = self.close_window()
        self.total_energy, self.total_seconds = allocate_block_energy(
            self.pending, joules, self.total_energy, self.total_seconds)
        self.write_csv()
        for row in self.pending:
            print(json.dumps(row), flush=True)
        self.pending.clear()

    def write_csv(self):
        with (self.output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def _csv(self):
        # Native code fills validation_accuracy immediately before this callback.
        if len(self.pending) >= 5:
            self.flush()

    def close(self):
        self.energy_counter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Baseline compatibility argument; GateZO uses internal L_max=49')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--max-hidden', type=int, default=49)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    native = native_settings(batch_size=args.batch_size, device=args.device, seed=args.seed)
    if min(args.epochs, args.batch_size, args.max_hidden) < 1:
        parser.error('epochs, batch-size and max-hidden must be positive')
    if args.max_hidden > native['L_max'] * native['nB']:
        parser.error('max-hidden cannot exceed L-max * nB')
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
    destination = args.output / args.dataset / ('smoke' if args.smoke else 'full') / 'gatezo' / f'seed_{args.seed}'
    config = dict(method='gatezo', max_hidden=args.max_hidden, layers=1, native_args=native,
                  source=str(BUNDLE), data_root=str(args.data_root.resolve()), smoke=args.smoke,
                  train_samples=len(x), test_samples=len(x_test), seed=args.seed,
                  epochs_unit='growth_loop', stopping_policy='native_L_max_with_neuron_target',
                  requested_epochs=args.epochs, epoch_limit_applied=False,
                  growth_round_limit=native['L_max'], growth_round_limit_applied=True,
                  candidate_failure_policy='stop_and_fail_if_target_unmet',
                  torch_version=torch.__version__, deterministic_algorithms=True,
                  cudnn_benchmark=False, cudnn_deterministic=True,
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  energy_measurement_version='nvml_total_growth_block_time_allocation_v1',
                  bp_memory_version=BP_MEMORY_VERSION,
                  auxiliary_memory_version=AUX_MEMORY_VERSION,
                  gate_persistent_state_policy='structure_in_W_U_Beta_no_learned_gate_state',
                  bp_memory_aggregation='no_backward_zero',
                  eval_each_epoch=False, energy_window_epochs=5,
                  accuracy_reporting_version='train_and_final_test_v1',
                  measurement_window='growth_search_gate_fit_and_selection_validation_excludes_final_evaluation')
    config.update(data_metadata(args, train))
    config['steps'] = x.shape[1]
    if skip_completed(destination, config):
        return
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    recorder = None
    try:
        model = BoundedGateZO(max_hidden=args.max_hidden, **native)
        recorder = Recorder(model, x, targets, x_test, t_test, destination)
        model._pp3_runtime = recorder
        model.classification(x, targets, x_test, t_test)
        if model.hidden_count != args.max_hidden:
            raise AssertionError('Final hidden size differs from target')
        torch.save(dict(W=model.W, U=model.U, Beta=model.Beta,
                        growth_round=int(model.L), hidden_neurons=model.hidden_count,
                        config=config), destination / 'model.pt')
        (destination / 'summary.json').write_text(json.dumps(model.gate_zo_summary, indent=2))
        (destination / 'status.json').write_text(json.dumps(dict(status='success', hidden_neurons=model.hidden_count)))
    except BaseException:
        if recorder is not None and recorder.memory is not None:
            recorder.finish_measurement()
        (destination / 'status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc())))
        raise
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == '__main__':
    main()
