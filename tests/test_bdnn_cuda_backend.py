import contextlib
import io
import sys
import unittest
import warnings
import json
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from constants import make_config
from bdnn_gpu.model import BDNNGPU
from pp3.flops import FlopCounter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'BDNNversion1-main'))
from bdnn1 import bdnn1


class CUDABackendTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required for entrypoint parity')
    def test_native_and_cuda_entry_stop_at_same_round_and_preserve_final_slice(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = make_config('bdnn', 'spike')
            config['data'].update(root=str(folder), steps=6, inputs=2, classes=2)
            config['model'].update(epochs=16, pool_size=4, growth_step=1,
                                   lambdas=[1, 3], r=[.99, .999])
            config.update(device='cuda:0', threads=1)
            for split in ('train', 'test'):
                np.save(folder/config['data'][split+'_x'], np.full((4, 6, 2), 3., dtype='float32'))
                np.save(folder/config['data'][split+'_y'], np.array([0, 1, 0, 1]))
            records = []
            for name, entry in [('native', 'BDNNversion1-main/main_exp1.py'), ('cuda', 'bdnn_gpu/main.py')]:
                output = folder/name
                config['output'] = str(output)
                env = dict(os.environ, PP3_CONFIG='environment', PP3_CONFIG_JSON=json.dumps(config),
                           PYTHONPATH=str(root))
                process = subprocess.run([sys.executable, str(root/entry)], cwd=root, env=env,
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
                self.assertEqual(process.returncode, 0, process.stdout[-2000:]+process.stderr[-2000:])
                status = json.loads((output/'status.json').read_text())
                self.assertTrue(status['training_completed'])
                if name == 'cuda':
                    self.assertTrue(status['flops_complete'])
                import csv
                with (output/'metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                records.append((status, rows, torch.load(output/'model.pt', weights_only=False)))
            native, cuda = records
            # Constant accuracy triggers the second native periodic check;
            # the round that performs that check still executes before stopping.
            self.assertEqual(native[0]['epochs'], 15)
            self.assertEqual(cuda[0]['epochs'], native[0]['epochs'])
            for before, after in zip(native[1], cuda[1]):
                self.assertEqual(before['train_accuracy'], after['train_accuracy'])
                self.assertEqual(before['test_accuracy'], after['test_accuracy'])
            for key in ('W', 'U', 'Beta'):
                np.testing.assert_allclose(native[2][key], cuda[2][key], atol=1e-6)
            self.assertEqual(cuda[0]['final_slice_index'], 0)
            self.assertEqual(cuda[0]['actual_device'], 'cuda:0')

    def test_controller_keeps_other_entries_and_isolates_default_output(self):
        root = Path(__file__).resolve().parents[1]
        args = ['--dry-run', '--datasets', 'spike', '--seeds', '57']
        old = json.loads(subprocess.check_output([sys.executable, str(root/'run_experiments.py'), *args], cwd=root))
        new = json.loads(subprocess.check_output([sys.executable, str(root/'run_experiments_bdnn_gpu.py'), *args], cwd=root))
        for before, after in zip(old['tasks'], new['tasks']):
            self.assertEqual(before['model'], after['model'])
            self.assertEqual(before['data'], after['data'])
            self.assertNotEqual(before['output'], after['output'])
            if before['method'] == 'bdnn':
                self.assertEqual(Path(after['entrypoint']).parent.name, 'bdnn_gpu')
                self.assertNotEqual(before['experiment_id'], after['experiment_id'])
            else:
                self.assertEqual(before['entrypoint'], after['entrypoint'])

    def test_legacy_class_count_failure_is_preserved(self):
        # Even many accepted candidates cannot satisfy nB=3 with C having 2 rows.
        x = np.full((4, 6, 2), 3., dtype='float32')
        target = np.eye(2, dtype='float32')[[0, 1, 0, 1]]
        settings = self.settings()
        settings['growth_step'] = 3
        for device in self.devices():
            np.random.seed(57)
            native = bdnn1(T_max=4, nB=3, Lambdas=[1], r=[.99, .999])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, _, failed = native.sc_Search(x, target)
            np.random.seed(57)
            model = BDNNGPU(2, 2, settings, device)
            found = model.sc_Search(torch.tensor(x, device=device), torch.tensor(target, device=device))
            self.assertEqual(failed, 1)
            self.assertIsNone(found)

    def settings(self):
        s = make_config('bdnn', 'spike')['model']
        s.update(pool_size=4, growth_step=1, lambdas=[1], r=[.99, .999])
        return s

    def test_second_growth_keeps_block_diag_and_complete_accounting(self):
        for device in self.devices():
            model = BDNNGPU(2, 2, self.settings(), device)
            w = torch.ones(2, 1, dtype=torch.float64, device=device)
            u = torch.ones(1, 1, dtype=torch.float64, device=device)
            model.counter = counter = FlopCounter()
            with counter:
                model.addNodes(w, u)
                model.addNodes(w, u)
                x = torch.full((4, 6, 2), 3., device=device)
                target = torch.eye(2, device=device)[torch.tensor([0, 1, 0, 1], device=device)]
                model.upgradeSCN(x, target)
            self.assertTrue(torch.equal(model.U, torch.eye(2, dtype=torch.float64, device=device)))
            self.assertEqual(model.W.dtype, torch.float32)
            self.assertTrue(counter.summary()['complete'], counter.summary()['unknown'])

    def devices(self):
        return ['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']

    def test_fixed_weights_match_native_recurrence_readout_and_loss(self):
        rng = np.random.RandomState(57)
        x = rng.uniform(0, 5, (7, 12, 2)).astype('float32')
        target = np.eye(2, dtype='float32')[[0, 1, 0, 1, 0, 1, 0]]
        w = rng.uniform(0, 3, (2, 2))
        u = rng.normal(0, .1, (2, 2))
        native = bdnn1()
        native.W, native.U = w, u
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            h, spikes = native.getH(x)
            output, loss = native.upgradeSCN(x, target)
            predicted = native.getOutput(x)
        for device in self.devices():
            model = BDNNGPU(2, 2, self.settings(), device)
            model.W = torch.tensor(w, device=device)
            model.U = torch.tensor(u, device=device)
            xx, yy = torch.tensor(x, device=device), torch.tensor(target, device=device)
            actual_h, actual_spikes = model.getH(xx)
            np.testing.assert_allclose(actual_h.cpu(), h, atol=6e-8, rtol=1e-7)
            np.testing.assert_array_equal(actual_spikes.cpu(), spikes)
            actual, actual_loss = model.upgradeSCN(xx, yy)
            np.testing.assert_allclose(actual.cpu(), output, atol=2e-5)
            np.testing.assert_allclose(actual_loss.cpu(), loss, atol=2e-5)
            np.testing.assert_allclose(model.getOutput(xx).cpu(), predicted, atol=2e-5)
            self.assertEqual(model.Beta.device, torch.device(device))

    def test_fixed_candidates_match_legacy_selection(self):
        rng = np.random.RandomState(57)
        uniform, normal = rng.rand(2, 4), rng.randn(4, 4)
        x = np.full((4, 6, 2), 3., dtype='float32')
        target = np.eye(2, dtype='float32')[[0, 1, 0, 1]]
        native = bdnn1(T_max=4, nB=1, Lambdas=[1], r=[.99, .999])
        with patch('numpy.random.rand', return_value=uniform), patch('numpy.random.randn', return_value=normal), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w, u, failed = native.sc_Search(x, target)
        self.assertEqual(failed, 0)
        for device in self.devices():
            model = BDNNGPU(2, 2, self.settings(), device)
            found = model.sc_Search(torch.tensor(x, device=device), torch.tensor(target, device=device),
                [(torch.tensor(2*uniform-1, device=device), torch.tensor(2*normal-1, device=device))])
            np.testing.assert_array_equal(found[0].cpu(), w)
            np.testing.assert_array_equal(found[1].cpu(), u)

    def test_solver_count_once_and_empty_rank_deficient_wide(self):
        for device in self.devices():
            for h in (torch.tensor([[1., 0.], [0., 1.], [1., 1.]]), torch.ones(2, 3), torch.empty(3, 0)):
                h = h.to(device)
                target = torch.ones(h.shape[0], 2, device=device)
                model = BDNNGPU(2, 2, self.settings(), device)
                counter = FlopCounter()
                model.counter = counter
                with patch('torch.linalg.pinv', wraps=torch.linalg.pinv) as pinv, counter:
                    model.computeBeta(h, target)
                self.assertEqual(pinv.call_count, 1)
                summary = counter.summary()
                self.assertTrue(summary['complete'], summary['unknown'])
                m, n = h.shape
                q, p = min(m, n), max(m, n)
                estimate = 0 if not q else 4*p*q*q+8*q**3+1+q+q*m+2*n*q*m
                self.assertEqual(summary['other_flops_estimated'], estimate)
                self.assertEqual(summary['known_flops'], estimate+2*n*m*2)

    def test_counting_does_not_change_device_random_stream_or_model(self):
        for device in self.devices():
            results = []
            for enabled in (False, True):
                torch.manual_seed(57)
                np.random.seed(57)
                model = BDNNGPU(2, 2, self.settings(), device)
                x = torch.full((4, 6, 2), 3., device=device)
                y = torch.eye(2, device=device)[torch.tensor([0, 1, 0, 1], device=device)]
                model.counter = counter = FlopCounter(enabled=enabled)
                with counter:
                    found = model.sc_Search(x, y)
                    self.assertIsNotNone(found)
                    model.addNodes(*found)
                    model.upgradeSCN(x, y)
                    prediction = model.getOutput(x)
                results.append((model.state_dict(), prediction))
            for key in results[0][0]:
                self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))
            self.assertTrue(torch.equal(results[0][1], results[1][1]))
            self.assertTrue(counter.summary()['complete'], counter.summary()['unknown'])


if __name__ == '__main__':
    unittest.main()
