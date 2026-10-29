"""Native source and full classification equivalence, including random streams."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import torch
from bdnn import NATIVE, ARTIFACTS, reuse
from native_bdnn_metrics import NativeMetrics

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', [str(path.parent), *sys.path]):
        spec.loader.exec_module(result)
    return result


class NativeBDNNTests(unittest.TestCase):
    def test_copied_core_and_settings_are_unchanged(self):
        for path in (ROOT/'BDNNversion1-main').glob('*.py'):
            if path.name != 'main_exp1.py':
                self.assertEqual((NATIVE/path.name).read_bytes(), path.read_bytes(), path.name)
        defaults = module(NATIVE/'config.py', 'native_defaults').SCN_DEFAULTS
        self.assertEqual((defaults['nB'], defaults['L_max'], defaults['T_max']), (3, 200, 100))

    def test_complete_native_classification_and_rng_parity(self):
        original = module(ROOT/'BDNNversion1-main/bdnn1.py', 'original_bdnn').bdnn1
        copied = module(NATIVE/'bdnn1.py', 'copied_bdnn').bdnn1
        defaults = module(NATIVE/'config.py', 'native_defaults').SCN_DEFAULTS
        x = np.random.RandomState(2).randint(0, 2, (24, 12, 9)).astype(np.float32)
        target = np.eye(12, dtype=np.float32)[np.arange(24)%12]
        runs = []
        with tempfile.TemporaryDirectory() as folder:
            for cls, observe in ((original, False), (copied, True)):
                np.random.seed(57)
                torch.manual_seed(57)
                model = cls(**defaults)
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if observe:
                        recorder = NativeMetrics(model, folder, 'cpu')
                        model._pp3_runtime = recorder
                    result = model.classification(x, target, x, target)
                    if observe:
                        recorder.finish(dict(W=result[4], U=result[5], Beta=result[6]))
                runs.append((result, model.L, np.random.get_state(), torch.random.get_rng_state()))
            for index in (0, 1, 2, 4, 5, 6):  # native wall-clock timeList is deliberately excluded
                np.testing.assert_array_equal(np.asarray(runs[0][0][index]), np.asarray(runs[1][0][index]))
            self.assertEqual(runs[0][1], runs[1][1])
            for a, b in zip(runs[0][2], runs[1][2]):
                np.testing.assert_array_equal(a, b)
            self.assertTrue(torch.equal(runs[0][3], runs[1][3]))
            self.assertTrue(all(row['test_acc'] is not None for row in recorder.rows[:-1]))
            self.assertEqual(recorder.rows[-1]['final_refit_and_test'], 1)
            self.assertEqual(recorder.rows[-1]['hidden_neurons'], np.asarray(result[4]).shape[1])

    def test_no_fixed_width_completion_requirement(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)/'native'
            config = dict(protocol='native_main_exp1_dataset_only_v1')
            self.assertFalse(reuse(destination, config))
            destination.mkdir()
            for name in ARTIFACTS:
                (destination/name).write_text('native artifact')
            (destination/'config.json').write_text(json.dumps(config))
            (destination/'status.json').write_text(json.dumps(dict(status='success')))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(reuse(destination, config))
            with self.assertRaises(FileExistsError):
                reuse(destination, dict(protocol='old_fixed_49'))

    def test_dataset_only_native_entry(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            x = np.random.RandomState(2).randint(0, 2, (24, 12, 9)).astype(np.float32)
            y = np.eye(12, dtype=np.float32)[np.arange(24)%12]
            np.savez(root/'texture_spikesT450_T77.npz', X_train=x, Y_train=y, X_test=x, Y_test=y)
            env = os.environ.copy()
            env.pop('PP3_CONFIG', None)
            env.pop('PP3_CONFIG_JSON', None)
            env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            result = subprocess.run([sys.executable, str(NATIVE.parent/'bdnn.py'),
                                    '--device', 'cpu', '--data-root', str(root), '--output', str(root/'out'),
                                    '--epochs', '1', '--batch-size', '1', '--seed', '999'], cwd=root,
                                    env=env, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace')[-3000:])
            destination = root/'out/texture-mini/full/bdnn/native'
            for name in ARTIFACTS:
                self.assertTrue((destination/name).is_file(), name)
            config = json.loads((destination/'config.json').read_text())
            self.assertEqual(config['native_args'], module(NATIVE/'config.py', 'defaults').SCN_DEFAULTS)
            self.assertIsNone(config['random_seed'])
            self.assertEqual(json.loads((destination/'status.json').read_text())['status'], 'success')


if __name__ == '__main__':
    unittest.main()
