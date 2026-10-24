"""Exercise five-epoch counter windows through the actual baseline loop."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from data import Sequences
from train import main, allocate_block_energy


class EnergyBlockTests(unittest.TestCase):
    def test_time_allocation_preserves_total_and_missing_data(self):
        rows = [dict(epoch_seconds=t) for t in (1., 2., 2., 1., 4.)]
        totals = allocate_block_energy(rows, 100., 0., 0.)
        self.assertEqual([r['energy_j'] for r in rows], [10., 20., 20., 10., 40.])
        self.assertEqual(totals, (100., 10.))
        self.assertTrue(all(r['power_w'] == 10. for r in rows))
        missing = [dict(epoch_seconds=1.)]
        self.assertEqual(allocate_block_energy(missing, None, *totals), (None, 11.))
        self.assertIsNone(missing[0]['energy_j'])

    def run_baseline(self, root, epochs, evaluate_enabled=False):
        dataset = Sequences(np.ones((4, 3, 9), dtype=np.float32), np.arange(4), 12)
        events = []
        counter = Mock()
        readings = iter([(0., 1000), (1., 16000), (2., 16000), (3., 22000),
                         (4., 22000), (5., 28000)])

        def read():
            events.append('counter')
            return next(readings)

        def evaluate(*args):
            self.assertTrue(evaluate_enabled, 'Evaluation must be disabled by default')
            events.append('evaluation')
            return dict(acc=.5, mse=.1)

        counter.read.side_effect = read
        argv = ['ann.py', '--device', 'cpu', '--epochs', str(epochs), '--batch-size', '4',
                '--data-root', str(root), '--output', str(root)]
        if evaluate_enabled:
            argv.append('--eval-each-epoch')
        with patch.object(sys, 'argv', argv), patch('train.load_data', return_value=(dataset, dataset)), \
             patch('train.data_metadata', return_value={}), patch('train.TotalEnergyCounter', return_value=counter), \
             patch('train.evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            main('ann')
        out = root/'texture-mini/full/ann/seed_57'
        with (out/'metrics.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        counter.close.assert_called_once()
        self.assertFalse((out/'metrics_metadata.jsonl').exists())
        return rows, events, json.loads((out/'config.json').read_text())

    def test_five_epoch_block_and_final_remainder_without_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, events, config = self.run_baseline(Path(directory), 7)
            self.assertEqual(events, ['counter'] * 4)
            self.assertEqual(len(rows), 7)
            self.assertEqual(config['energy_window_epochs'], 5)
            self.assertFalse(config['eval_each_epoch'])
            self.assertNotIn('test_acc', rows[0])
            self.assertNotIn('train_acc', rows[0])
            self.assertIn('hidden_state_mem_bytes (included)', rows[0])
            energies = [float(row['energy_j']) for row in rows]
            self.assertAlmostEqual(sum(energies[:5]), 15.)
            self.assertAlmostEqual(sum(energies[5:]), 6.)
            self.assertAlmostEqual(float(rows[-1]['total_energy_j']), 21.)
            for block in (rows[:5], rows[5:]):
                powers = [float(row['energy_j']) / float(row['epoch_seconds']) for row in block]
                self.assertTrue(all(abs(powers[0] - p) < 1e-6 for p in powers))

    def test_enabled_evaluation_occurs_after_each_closed_counter_window(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, events, config = self.run_baseline(Path(directory), 3, True)
            self.assertEqual(events, ['counter', 'counter', 'evaluation', 'evaluation'] * 3)
            self.assertEqual(config['energy_window_epochs'], 1)
            self.assertTrue(all(float(row['test_acc']) == .5 for row in rows))


if __name__ == '__main__':
    unittest.main()
