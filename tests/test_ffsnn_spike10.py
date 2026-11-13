import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from constants import make_config
from ffsnn_spike10.runtime import WindowSumDataset


ROOT = Path(__file__).resolve().parents[1]


class SpikeTenTests(unittest.TestCase):
    def test_window_boundaries_full_coverage_and_labels(self):
        x = torch.zeros(2, 400, 64)
        # Impulses at every boundary and at the final source frame expose
        # truncation, subsampling, shifted windows and incorrect axis handling.
        for t in range(400):
            x[0, t, t % 64] = 1
        x[1, 39, 0], x[1, 40, 0], x[1, 399, 0] = 2, 3, 4
        ds = WindowSumDataset(TensorDataset(x, torch.tensor([7, 19])))
        for i in range(2):
            result, label = ds[i]
            expected = torch.stack([x[i, start:start+40].sum(0) for start in range(0, 400, 40)])
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(result.sum(0), x[i].sum(0))
            self.assertEqual(result.shape, (10, 64))
            self.assertEqual(int(label), [7, 19][i])
        self.assertEqual(float(ds[1][0][9, 0]), 4.)

    def test_worker_processes_receive_same_transform(self):
        x = torch.ones(4, 400, 64)
        dataset = WindowSumDataset(TensorDataset(x, torch.arange(4)))
        for inputs, labels in DataLoader(dataset, batch_size=2, num_workers=2):
            self.assertEqual(inputs.shape, (2, 10, 64))
            self.assertTrue(bool((inputs == 40).all()))

    def test_reject_subsampled_source(self):
        dataset = WindowSumDataset(TensorDataset(torch.zeros(1, 16, 64), torch.zeros(1)))
        with self.assertRaises(ValueError):
            dataset[0]

    def test_training_copy_has_only_adapter_changes(self):
        native = (ROOT/'SNN_FF-main/FF_SNN/SNNFF_SHD.py').read_text(encoding='utf-8')
        copied = (ROOT/'ffsnn_spike10/SNNFF_SPIKE10.py').read_text(encoding='utf-8')
        copied = copied.replace('from ffsnn_spike10.runtime import from_environment',
                                'from pp3.native_runtime import from_environment')
        copied = copied.replace("num_steps = settings['time_steps']  # 10 windows, each summing 40 source steps",
                                "num_steps = runtime.config.get('smoke_steps') or spec['steps']")
        self.assertEqual(copied, native)

    def test_controller_isolates_method_and_preserves_original_config(self):
        for smoke in ([], ['--smoke']):
            output = subprocess.check_output([sys.executable, str(ROOT/'run_experiments_ffsnn10.py'),
                '--dry-run', '--seeds', '57', '--output', str(ROOT/'outputs/shared_test_root'), *smoke], cwd=ROOT)
            task, = json.loads(output)['tasks']
            self.assertEqual(task['method'], 'ffsnn10')
            self.assertEqual(task['dataset'], 'spike')
            self.assertEqual(task['data']['steps'], 400)
            self.assertEqual(task['model']['time_steps'], 10)
            self.assertEqual(task['model']['layers'], 2)
            self.assertEqual(Path(task['output']).parts[-3:], ('spike', 'ffsnn10', 'seed_57'))
            self.assertIsNone(task.get('smoke_steps'))
        native = make_config('ffsnn', 'spike')
        self.assertEqual(native['data']['steps'], 400)
        self.assertNotIn('time_steps', native['model'])


if __name__ == '__main__':
    unittest.main()
