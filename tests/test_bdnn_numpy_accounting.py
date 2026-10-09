import contextlib
import io
import csv
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'BDNNversion1-main'))
from bdnn1 import bdnn1
import numpy_flops
from standardmatrxi import softmax_standardization
from sklearn import metrics
from pp3.flops import FlopCounter


class BDNNNumpyAccountingTests(unittest.TestCase):
    def assert_rng_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertEqual(left[2:], right[2:])

    def test_numpy_matmul_and_torch_matmul_are_counted_once_each(self):
        a, b = np.ones((2, 3)), np.ones((3, 4))
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        counter = FlopCounter()
        with counter:
            actual = numpy_flops.matmul(counter, a, b)
            tensor = ta @ tb
        np.testing.assert_array_equal(actual, tensor.numpy())
        self.assertEqual(counter.summary()['flops'], 96)
        rows = counter.summary()['operations']
        self.assertEqual(sum(r['flops'] for r in rows if r['operator'].startswith('manual::')), 48)

    def test_mixed_scale_checks_dispatch_not_output_type(self):
        # Current Tensor * ndarray returns a Tensor but does NumPy multiply.
        # Tensor * Tensor, including an explicit array conversion, uses ATen.
        counter = FlopCounter()
        scalar, array = torch.tensor(2.), np.ones((2, 3))
        with counter, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = numpy_flops.multiply(counter, scalar, array)
            tensor = numpy_flops.multiply(counter, scalar, torch.from_numpy(array))
        np.testing.assert_array_equal(np.asarray(result), tensor.numpy())
        self.assertEqual(counter.summary()['flops'], 12)
        self.assertFalse(counter.summary()['unknown'])

    def test_inequality_matches_original_and_does_not_duplicate_torch_subtraction(self):
        eq = np.array([[1.], [2.], [3.]])
        gk = np.array([[2.], [1.], [4.]])
        for r in (torch.tensor(.9), .9):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a = (1-r) * eq.conj().T
                expected = (eq.conj().T @ gk)**2 / (gk.conj().T @ gk) - np.array(a) @ eq
                counter = FlopCounter()
                model = bdnn1()
                model._pp3_runtime = SimpleNamespace(counter=counter)
                with counter:
                    actual = model.inequalityEq(eq, gk, r)
            np.testing.assert_array_equal(actual, expected)
            # 3 dot products at 6 each + 3 scale multiplies + scalar
            # subtraction, square, divide, subtraction = 25.
            self.assertEqual(counter.summary()['flops'], 25)
            self.assertEqual(sum(x['calls'] for x in counter.summary()['operations']
                                 if x['operator'] == 'manual::numpy::matmul'), 3)

    def test_pinv_single_solver_call_estimate_and_beta_product(self):
        matrix = np.array([[1., 0.], [0., 1.], [1., 1.]])
        target = np.arange(12., dtype=float).reshape(3, 4)
        expected = np.linalg.pinv(matrix) @ target
        model = bdnn1()
        counter = FlopCounter()
        model._pp3_runtime = SimpleNamespace(counter=counter)
        with patch.object(np.linalg, 'pinv', wraps=np.linalg.pinv) as solver:
            with counter:
                model.computeBeta(matrix, target)
            self.assertEqual(solver.call_count, 1)
        np.testing.assert_array_equal(model.Beta, expected)
        self.assertEqual(counter.summary()['other_flops_estimated'], 145)
        self.assertEqual(counter.summary()['flops'], 193)

    def test_pinv_wide_rank_deficient_and_empty(self):
        for matrix, cost in ((np.ones((2, 3)), 143), (np.empty((3, 0)), 0)):
            counter = FlopCounter()
            expected = np.linalg.pinv(matrix)
            with counter:
                actual = numpy_flops.pinv(counter, matrix)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(counter.summary()['flops'], cost)
        counter = FlopCounter()
        with counter:
            numpy_flops.pinv(counter, np.eye(2, dtype=complex))
        self.assertFalse(counter.summary()['complete'])

    def test_softmax_and_metrics_hand_calculated(self):
        matrix = np.array([[1., 2., 3.], [0., 1., 0.]])
        target = np.array([[0., 0., 1.], [0., 1., 0.]])
        counter = FlopCounter()
        with counter:
            output = numpy_flops.softmax(counter, softmax_standardization, matrix)
            mse = numpy_flops.metric(counter, 'mean_squared_error', metrics.mean_squared_error, target, output)
            accuracy = numpy_flops.metric(counter, 'accuracy_score', metrics.accuracy_score, target, target)
        np.testing.assert_array_equal(output, softmax_standardization(matrix))
        self.assertEqual(mse, metrics.mean_squared_error(target, output))
        self.assertEqual(accuracy, 1.)
        self.assertEqual(counter.summary()['flops'], 22 + 21 + 2)

    def test_metric_does_not_overlay_tensor_backend(self):
        counter = FlopCounter()
        x = torch.ones(3)
        with counter:
            numpy_flops.metric(counter, 'accuracy_score', lambda a, b: (a+b).mean(), x, x)
        self.assertEqual(counter.summary()['known_flops'], 6)
        self.assertFalse(counter.summary()['complete'])
        self.assertFalse(any(r['operator'].startswith('manual::') for r in counter.summary()['operations']))

    def test_gaussian_cache_and_random_stream_unchanged(self):
        for prefill in (0, 1):
            np.random.seed(57)
            if prefill:
                np.random.randn(1)
            initial = np.random.get_state()
            expected = np.random.randn(3)
            final = np.random.get_state()
            np.random.set_state(initial)
            counter = FlopCounter()
            with counter:
                actual = numpy_flops.randn(counter, 3)
            np.testing.assert_array_equal(actual, expected)
            self.assert_rng_equal(np.random.get_state(), final)
            self.assertEqual(counter.summary()['other_flops_estimated'], 15 if prefill else 30)

    def test_disabled_paused_and_stages(self):
        counter = FlopCounter(enabled=False)
        counter.add_manual('disabled', 9)
        with counter:
            numpy_flops.matmul(counter, np.ones((2, 3)), np.ones((3, 4)))
        self.assertEqual(counter.summary()['known_flops'], 0)
        counter.enabled = True
        counter.paused = True
        counter.add_manual('paused', 9)
        numpy_flops.record(counter, 'paused', 9)
        self.assertFalse(counter.records)
        counter.paused = False
        for stage in ('train', 'test'):
            with counter.section(stage):
                numpy_flops.matmul(counter, np.ones((2, 3)), np.ones((3, 4)))
        self.assertEqual(counter.summary()['known_flops_by_stage'], dict(train=48, test=48))

    def test_real_candidate_search_counts_actual_calls_and_preserves_results(self):
        outputs = []
        x = np.full((4, 6, 2), 3., dtype=np.float32)
        target = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
        for enabled in (False, True):
            np.random.seed(57)
            torch.manual_seed(57)
            model = bdnn1(T_max=4, nB=1, Lambdas=[1, 3], r=[.99, .999])
            counter = FlopCounter(enabled=enabled)
            model._pp3_runtime = SimpleNamespace(counter=counter)
            with counter, warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter('ignore')
                result = model.sc_Search(x, target)
                if not result[2]:
                    model.addNodes(result[0], result[1])
                    model.upgradeSCN(x, target)
                    predictions = model.getOutput(x)
                else:
                    self.fail('Fixture must exercise a successful growth/readout')
            outputs.append((result, model.Beta.copy(), predictions, np.random.get_state(), torch.random.get_rng_state()))
        for a, b in zip(outputs[0][0], outputs[1][0]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for index in (1, 2):
            np.testing.assert_array_equal(outputs[0][index], outputs[1][index])
        self.assert_rng_equal(outputs[0][3], outputs[1][3])
        self.assertTrue(torch.equal(outputs[0][4], outputs[1][4]))
        self.assertTrue(counter.summary()['complete'], counter.summary()['unknown'])
        rows = counter.summary()['operations']
        affine = next(r for r in rows if r['operator'] == 'manual::numpy::candidate_uniform_affine')
        self.assertLessEqual(affine['calls'], 2)
        dots = sum(r['calls'] for r in rows if r['operator'] == 'manual::numpy::inequality_elementwise')
        self.assertEqual(dots % (4*2), 0)
        # Three dot products per candidate/target plus the single readout product.
        self.assertEqual(sum(r['calls'] for r in rows if r['operator'] == 'manual::numpy::matmul'), 3*dots+1)

    def test_native_entry_outputs_and_phase_totals(self):
        from constants import make_config
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = make_config('bdnn', 'spike')
            config['data'].update(root=str(directory), steps=6, inputs=2, classes=2)
            config['model'].update(epochs=1, pool_size=4, growth_step=1,
                                   lambdas=[1, 3], r=[.99, .999])
            config.update(device='cpu', threads=1)
            for split in ('train', 'test'):
                np.save(directory / config['data'][split+'_x'], np.full((4, 6, 2), 3., dtype=np.float32))
                np.save(directory / config['data'][split+'_y'], np.array([0, 1, 0, 1]))
            runs = []
            for enabled in (False, True):
                output = directory / str(enabled)
                config.update(output=str(output), count_flops=enabled)
                launch = directory / (str(enabled)+'.json')
                launch.write_text(json.dumps(config), encoding='utf-8')
                env = os.environ.copy()
                env.update(PP3_CONFIG=str(launch), PYTHONPATH=str(root))
                if enabled:
                    env.update(PP3_CONFIG='environment', PP3_CONFIG_JSON=json.dumps(config))
                probe = (
                    "import runpy,sys,json,os; from pathlib import Path; "
                    "from pp3.native_runtime import Runtime; phases={}; original=Runtime.end; "
                    "exec('def capture(self,*a,**kw):\\n r=original(self,*a,**kw)\\n phases[f\"epoch_{self.epoch:04d}_{self.phase}_operations.json\"]=r\\n return r'); "
                    "Runtime.end=capture; sys.path.insert(0,str(Path(sys.argv[1]).parent)); "
                    "runpy.run_path(sys.argv[1],run_name='__main__'); "
                    "out=Path(json.loads(os.environ.get('PP3_CONFIG_JSON') or Path(os.environ['PP3_CONFIG']).read_text())['output']); "
                    "(out/'test_phases.json').write_text(json.dumps(phases))"
                )
                proc = subprocess.run([sys.executable, '-c', probe, str(root/'BDNNversion1-main/main_exp1.py')],
                                      env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr[-5000:])
                status = json.loads((output/'status.json').read_text())
                self.assertTrue(status['training_completed'])
                self.assertFalse(list(output.glob('*_operations.json')))
                for name in ('config.json','environment.json','data.json','launch.json','array_preparation.json'):
                    self.assertFalse((output/name).exists(), name)
                phases = json.loads((output/'test_phases.json').read_text())
                for phase in phases.values():
                    self.assertEqual(sum(r['flops'] for r in phase['operations']), phase['known_flops'])
                    self.assertEqual(sum(phase['known_flops_by_stage'].values()), phase['known_flops'])
                    if not enabled:
                        self.assertEqual(phase['known_flops'], 0)
                if enabled:
                    training = phases['epoch_0001_train_operations.json']
                    self.assertGreater(training['known_flops'], 0)
                    self.assertEqual(training['flops'], training['known_flops'])
                    self.assertTrue(training['complete'], training['unknown'])
                    self.assertTrue(training['excluded'])
                    self.assertTrue(phases['epoch_0001_test_operations.json']['complete'])
                    self.assertTrue(phases['epoch_0000_final_refit_operations.json']['complete'])
                state = torch.load(output/'model.pt', weights_only=False)
                with (output/'metrics.csv').open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                runs.append((state, [(r['train_accuracy'], r['test_accuracy'], r['train_loss']) for r in rows]))
            for key in runs[0][0]:
                np.testing.assert_array_equal(np.asarray(runs[0][0][key]), np.asarray(runs[1][0][key]))
            self.assertEqual(runs[0][1], runs[1][1])


if __name__ == '__main__':
    unittest.main()
