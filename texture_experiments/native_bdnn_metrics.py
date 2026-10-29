"""Observe native BDNN callbacks; never evaluate or modify the model."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from energy import TotalEnergyCounter


def parameter_bytes(model):
    return sum(np.asarray(value).nbytes for value in (model.W, model.U, model.Beta))


class NativeMetrics:
    counter = None

    def __init__(self, model, output, device):
        self.model, self.output = model, Path(output)
        self.meter = TotalEnergyCounter(torch.device(device))
        self.rows, self.results = [], {}
        self.epoch = 0
        self.total_energy, self.total_seconds = 0., 0.

    def begin(self, phase, epoch=None, **unused):
        self.phase = phase
        if epoch is not None:
            self.epoch = epoch
        _, self.initial = self.meter.read()
        self.started = time.perf_counter()

    def end(self, accuracy=None, loss=None):
        _, final = self.meter.read()
        seconds = time.perf_counter()-self.started
        energy = (final-self.initial)/1000 if final is not None and self.initial is not None else None
        self.results[self.phase] = dict(seconds=seconds, energy=energy,
            accuracy=None if accuracy is None else float(accuracy), loss=None if loss is None else float(loss))

    def epoch_row(self, epoch, unit='growth_round'):
        self.append(epoch, self.results.get('train', {}), self.results.get('test', {}), False)
        self.results.clear()

    def append(self, epoch, training, testing, final):
        seconds, energy = training.get('seconds', 0.), training.get('energy')
        self.total_seconds += seconds
        self.total_energy = self.total_energy+energy if self.total_energy is not None and energy is not None else None
        row = dict(epoch=epoch, hidden_neurons=0 if isinstance(self.model.W, list) else self.model.W.shape[1],
            para_mem_bytes=parameter_bytes(self.model), optimizer_state_bytes=0, auxiliary_mem_bytes=0,
            **{'hiddenstatememfor bp': 0},
            train_acc=None if final else training.get('accuracy'), train_loss=training.get('loss'),
            test_acc=None if final else testing.get('accuracy'),
            final_test_acc=training.get('accuracy') if final else None,
            energy_j=energy, power_w=energy/seconds if energy is not None and seconds else None,
            total_energy_j=self.total_energy,
            total_power_w=self.total_energy/self.total_seconds if self.total_energy is not None and self.total_seconds else None,
            epoch_seconds=seconds, total_seconds=self.total_seconds,
            test_energy_j=testing.get('energy'), test_seconds=testing.get('seconds'),
            final_refit_and_test=int(final))
        self.rows.append(row)
        with (self.output/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self, state):
        try:
            self.append(self.epoch, self.results.get('final_refit', {}), {}, True)
            torch.save(state, self.output/'model.pt')
            (self.output/'final_metrics.json').write_text(json.dumps(self.rows[-1], indent=2), encoding='utf-8')
            (self.output/'summary.json').write_text(json.dumps(dict(
                protocol='native_main_exp1_dataset_only_v1',
                growth_rounds=int(self.model.L), retained_hidden_neurons=self.rows[-1]['hidden_neurons'],
                selection='native_test_accuracy_argmax_slice',
                final_measurement='native_selection_refit_and_final_test_combined',
                compute_device='cpu')), encoding='utf-8')
        finally:
            self.meter.close()
