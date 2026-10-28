"""Actual adapted growth: five-round windows and no intermediate test forward."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from gatezo import BDNN1GateZO, BoundedGateZO, Recorder


class GateZOEnergyBlockTests(unittest.TestCase):
    def run_growth(self, directory, hidden, failures, readings):
        model = BoundedGateZO(max_hidden=hidden, compute_device='cpu', nB=1,
            L_max=hidden, growth_protocol='validation_checkpoint', gate_zo_seed=22,
            gate_zo_verbose=False)
        x = np.ones((20, 2, 9), dtype=np.float32)
        labels = np.eye(2, dtype=np.float32)[np.arange(20) % 2]
        test_x = np.full((6, 2, 9), 99., dtype=np.float32)
        test_y = np.eye(2, dtype=np.float32)[np.arange(6) % 2]
        meter = Mock()
        meter.read.side_effect = [(float(i), value) for i, value in enumerate(readings)]
        attempted, test_calls, validation_calls = [], [], []

        def proposal(instance, data, targets):
            attempted.append(1)
            if len(attempted) <= failures:
                return [], [], 1
            return np.ones((9, 1)), np.zeros((1, 1)), 0

        def features(instance, data):
            if np.all(data == 99):
                test_calls.append(meter.read.call_count)
                self.assertFalse(recorder.window_open, 'Final test must be outside energy window')
                self.assertIsNone(recorder.memory)
                self.assertEqual(instance.hidden_count, hidden)
            elif len(data) < len(x):
                validation_calls.append(len(data))
            return np.ones((len(data), instance.hidden_count), dtype=np.float32), None

        with patch('gatezo.TotalEnergyCounter', return_value=meter), \
             patch.object(BDNN1GateZO, 'sc_Search', proposal), \
             patch.object(BoundedGateZO, 'getH', features), \
             patch.object(model.structural_gate, 'select',
                side_effect=lambda m, x, t, old, proposed: {'kept': proposed}), \
             contextlib.redirect_stdout(io.StringIO()):
            recorder = Recorder(model, x, labels, test_x, test_y, directory)
            model._pp3_runtime = recorder
            try:
                model.classification(x, labels, test_x, test_y)
            finally:
                recorder.close()
        meter.close.assert_called_once()
        self.assertEqual(meter.read.call_count, len(readings))
        self.assertEqual(test_calls, [len(readings)])
        self.assertGreater(len(validation_calls), 0)
        with (directory/'metrics.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), hidden + failures)
        for row in rows:
            self.assertNotIn('test_acc', row)
            self.assertNotIn('train_acc', row)
            self.assertEqual(int(row['hiddenstatememfor bp']), 0)
            self.assertEqual(int(row['optimizer_state_bytes']), 0)
            self.assertEqual(int(row['auxiliary_mem_bytes']), 0)
        final = json.loads((directory/'final_metrics.json').read_text())
        self.assertEqual(final['hiddenstatememfor bp'], 0)
        self.assertEqual(final['auxiliary_mem_bytes'], 0)
        return rows, final

    def test_five_rounds_remainder_and_final_test(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, final = self.run_growth(Path(folder), 7, 0,
                [1000, 16000, 16000, 22000, 22000, 25000])
        for group, total in ((rows[:5], 15.), (rows[5:], 6.)):
            self.assertAlmostEqual(sum(float(r['energy_j']) for r in group), total)
            duration = sum(float(r['epoch_seconds']) for r in group)
            for row in group:
                self.assertAlmostEqual(float(row['energy_j']),
                                       total * float(row['epoch_seconds']) / duration)
        self.assertTrue(all(r['validation_accuracy'] != '' for r in rows))
        self.assertEqual(final['energy_j'], 3.)
        self.assertEqual(final['total_energy_j'], 24.)
        self.assertIn('test_acc', final)

    def test_exact_five_rounds_missing_energy_stays_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, final = self.run_growth(Path(folder), 5, 0, [None, None, None, None])
        self.assertTrue(all(row['energy_j'] == '' for row in rows))
        self.assertIsNone(final['energy_j'])
        self.assertIsNone(final['total_energy_j'])


if __name__ == '__main__':
    unittest.main()
