"""Compare independently loaded spec metrics with the actual Texture adapter."""
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from gatezo import classification_metrics, target_growth


path = Path(__file__).parent / 'texture_bdnn_gatezo_spec/gate_zo_bdnn/structural_gate.py'
module_spec = importlib.util.spec_from_file_location('_texture_metric_spec', path)
spec = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = spec
module_spec.loader.exec_module(spec)


class MetricParityTests(unittest.TestCase):
    def test_same_inputs_match_spec_exactly(self):
        rng = np.random.default_rng(57)
        for dtype in (np.float32, np.float64):
            for n, classes, scale in ((1, 2, 1), (23, 12, 1), (17, 20, 1000)):
                with self.subTest(dtype=dtype, n=n, classes=classes, scale=scale):
                    features = rng.normal(size=(n, 7)).astype(dtype)
                    beta = (rng.normal(size=(7, classes)) * scale).astype(dtype)
                    targets = np.eye(classes, dtype=dtype)[rng.integers(classes, size=n)]
                    actual = classification_metrics(features, targets, beta)
                    self.assertEqual(actual, spec.classification_metrics(features, targets, beta))
                    # Independent PyTorch cross-check of formulas and denominators.
                    logits = torch.from_numpy(features @ beta).double()
                    labels = torch.from_numpy(targets.argmax(axis=1))
                    expected_loss = torch.nn.functional.cross_entropy(logits, labels).item()
                    expected_mse = ((logits.softmax(1) - torch.from_numpy(targets)) ** 2).mean().item()
                    np.testing.assert_allclose(actual['loss'], expected_loss, rtol=2e-6, atol=1e-10)
                    np.testing.assert_allclose(actual['mse'], expected_mse, rtol=2e-6, atol=1e-10)
                    self.assertEqual(actual['accuracy'], (logits.argmax(1) == labels).double().mean().item())

    def test_tied_logits_have_same_accuracy_and_distinct_ce_mse(self):
        features = np.zeros((4, 2))
        targets = np.eye(2)[[0, 1, 0, 1]]
        result = classification_metrics(features, targets, np.zeros((2, 2)))
        self.assertEqual(result, spec.classification_metrics(features, targets, np.zeros((2, 2))))
        self.assertEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['loss'], np.log(2))
        self.assertAlmostEqual(result['mse'], 0.25)

    def test_actual_final_refit_adapter_matches_spec(self):
        # Exercise the adapter's own refit function with telemetry enabled.
        adapted = inspect.getclosurevars(target_growth).nonlocals['adapted']
        refit = adapted.__globals__['refit_and_measure']
        rng = np.random.default_rng(91)
        train = rng.normal(size=(24, 5))
        test = rng.normal(size=(9, 5))
        targets = np.eye(3)[np.arange(24) % 3]
        test_targets = np.eye(3)[np.arange(9) % 3]
        for ridge in (0., 0.1):
            with self.subTest(ridge=ridge):
                runtime = SimpleNamespace(counter=None, finish_measurement=Mock())
                actual_model = SimpleNamespace(getH=lambda x: (x, None), _pp3_runtime=runtime)
                spec_model = SimpleNamespace(getH=lambda x: (x, None))
                actual = refit(actual_model, train, targets, test, test_targets, ridge)
                expected = spec.refit_and_measure(spec_model, train, targets, test, test_targets, ridge)
                np.testing.assert_array_equal(actual_model.Beta, spec_model.Beta)
                self.assertEqual(actual, expected)
                self.assertEqual(runtime.final_evaluation, expected)
                runtime.finish_measurement.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
