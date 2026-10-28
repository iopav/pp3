"""Exercise method-specific counter windows through the actual baseline loop."""
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

    def run_baseline(self, root, epochs, evaluate_enabled=False, method='ann'):
        dataset = Sequences(np.ones((4, 3, 9), dtype=np.float32), np.arange(4), 12)
        test_dataset = Sequences(np.zeros((2, 3, 9), dtype=np.float32), np.arange(2), 12)
        events = []
        counter = Mock()
        readings = iter([(0., 1000), (1., 16000), (2., 16000), (3., 22000),
                         (4., 22000), (5., 28000)])

        def read():
            events.append('counter')
            return next(readings)

        def evaluate(*args):
            self.assertIs(args[1].dataset, test_dataset, 'Training set must not be evaluated again')
            events.append('evaluation')
            return dict(acc=.5, mse=.1)

        counter.read.side_effect = read
        argv = ['ann.py', '--device', 'cpu', '--epochs', str(epochs), '--batch-size', '4',
                '--data-root', str(root), '--output', str(root)]
        if evaluate_enabled:
            argv.append('--eval-each-epoch')
        with patch.object(sys, 'argv', argv), patch('train.load_data', return_value=(dataset, test_dataset)), \
             patch('train.data_metadata', return_value={}), patch('train.TotalEnergyCounter', return_value=counter), \
             patch('train.evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            main(method)
        out = root/f'texture-mini/full/{method}/seed_57'
        with (out/'metrics.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        counter.close.assert_called_once()
        self.assertFalse((out/'metrics_metadata.jsonl').exists())
        import torch
        saved = torch.load(out/'model.pt', weights_only=False)
        self.assertEqual(saved['completed_epochs'], epochs)
        self.assertEqual(saved['optimizer']['param_groups'][0]['momentum'], .9)
        self.assertTrue(saved['optimizer']['state'])
        for row in rows:
            self.assertEqual(int(row['auxiliary_mem_bytes']),
                int(row['optimizer_state_bytes']) + int(row['hiddenstatememfor bp']))
            self.assertEqual(int(row['optimizer_state_bytes']), int(row['para_mem_bytes']))
        return rows, events, json.loads((out/'config.json').read_text())

    def test_five_epoch_block_and_final_remainder_without_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, events, config = self.run_baseline(Path(directory), 7, method='rnn')
            self.assertEqual(events, ['counter'] * 4 + ['evaluation'])
            self.assertEqual(len(rows), 7)
            self.assertEqual(config['energy_window_epochs'], 5)
            self.assertFalse(config['eval_each_epoch'])
            self.assertNotIn('test_acc', rows[0])
            self.assertTrue(all(0 <= float(r['train_acc']) <= 1 for r in rows))
            self.assertTrue(all(r['final_test_acc'] == '' for r in rows[:-1]))
            self.assertEqual(float(rows[-1]['final_test_acc']), .5)
            self.assertIn('hiddenstatememfor bp', rows[0])
            self.assertTrue(all(int(row['hiddenstatememfor bp']) > 0 for row in rows))
            energies = [float(row['energy_j']) for row in rows]
            self.assertAlmostEqual(sum(energies[:5]), 15.)
            self.assertAlmostEqual(sum(energies[5:]), 6.)
            self.assertAlmostEqual(float(rows[-1]['total_energy_j']), 21.)
            for block in (rows[:5], rows[5:]):
                powers = [float(row['energy_j']) / float(row['epoch_seconds']) for row in block]
                self.assertTrue(all(abs(powers[0] - p) < 1e-6 for p in powers))

    def test_ann_ten_epoch_window_and_remainder(self):
        for epochs in (1, 10, 13):
            with self.subTest(epochs=epochs), tempfile.TemporaryDirectory() as directory:
                rows, events, config = self.run_baseline(Path(directory), epochs)
                self.assertEqual(config['energy_window_epochs'], 10)
                self.assertEqual(events, ['counter'] * (2 if epochs <= 10 else 4) + ['evaluation'])
                self.assertEqual(len(rows), epochs)
                self.assertNotIn('test_acc', rows[0])
                self.assertAlmostEqual(sum(float(r['energy_j']) for r in rows[:10]), 15.)
                if epochs > 10:
                    self.assertAlmostEqual(sum(float(r['energy_j']) for r in rows[10:]), 6.)
                for block in (rows[:10], rows[10:]):
                    duration = sum(float(r['epoch_seconds']) for r in block)
                    energy = sum(float(r['energy_j']) for r in block)
                    for row in block:
                        self.assertAlmostEqual(float(row['energy_j']),
                                               energy * float(row['epoch_seconds']) / duration)

    def test_legacy_flag_does_not_add_training_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, events, config = self.run_baseline(Path(directory), 3, True)
            self.assertEqual(events, ['counter', 'counter'] * 3 + ['evaluation'])
            self.assertEqual(config['energy_window_epochs'], 1)
            self.assertTrue(all(0 <= float(row['train_acc']) <= 1 for row in rows))
            self.assertEqual([r['final_test_acc'] for r in rows], ['', '', '0.5'])
            self.assertNotIn('mse', rows[0])
            self.assertNotIn('test_acc', rows[0])


if __name__ == '__main__':
    unittest.main()
