"""Regression checks for the actual-neuron limit and runtime isolation."""
import sys
import inspect
import unittest
from unittest.mock import patch
import numpy as np
from gatezo import BDNN1GateZO, BoundedGateZO
from gate_zo_bdnn.structural_gate import run_gatezo_growth


class GateZOTests(unittest.TestCase):
    def test_retries_failures_ignores_early_stops_and_keeps_target_checkpoint(self):
        original_source = inspect.getsource(run_gatezo_growth)
        original_code = run_gatezo_growth.__code__
        model = BoundedGateZO(max_hidden=7, compute_device='cpu', nB=3,
                              L_max=1, tol=1e9, verbose=1,
                              growth_protocol='validation_checkpoint', gate_zo_seed=22,
                              gate_zo_verbose=False)
        x = np.ones((20, 1, 9), dtype=np.float32)
        targets = np.eye(2, dtype=np.float32)[np.arange(20) % 2]
        calls = []
        def proposal(instance, x, targets):
            calls.append(instance.nB)
            if len(calls) <= 2:
                return [], [], 1
            return np.ones((9, instance.nB)), np.zeros((instance.nB, instance.nB)), 0
        def hidden(instance, data):
            return np.ones((len(data), model.hidden_count), dtype=np.float32), None
        # Flat accuracy and tiny error would stop the native loop, while its
        # historical best checkpoint would have only three neurons.
        with patch.object(BDNN1GateZO, 'sc_Search', proposal), \
             patch.object(BoundedGateZO, 'getH', hidden), \
             patch.object(model.structural_gate, 'select',
                          side_effect=lambda m, x, t, old, proposed: {'kept': proposed}):
            model.classification(x, targets, x, targets)
        self.assertEqual(calls, [3, 3, 3, 3, 1])
        self.assertEqual(model.hidden_count, 7)
        self.assertEqual(model.gate_zo_summary['attempted_rounds'], 5)
        self.assertEqual(model.gate_zo_summary['hidden_size'], 7)
        self.assertIs(run_gatezo_growth.__code__, original_code)
        self.assertEqual(inspect.getsource(run_gatezo_growth), original_source)

    def test_remaining_capacity_bounds_block_and_stops_at_49(self):
        model = BoundedGateZO(max_hidden=49, compute_device='cpu', nB=3)
        model.W = np.zeros((9, 48), dtype=np.float32)
        model.U = np.zeros((48, 48), dtype=np.float32)
        def proposal(instance, x, targets):
            self.assertEqual(instance.nB, 1)
            return np.zeros((9, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), 0
        with patch.object(BDNN1GateZO, 'sc_Search', proposal):
            w, u, flag = model.sc_Search(None, None)
            self.assertEqual(flag, 0)
            model.addNodes(w, u)
        self.assertEqual(model.hidden_count, 49)
        self.assertEqual(model.nB, 3)
        self.assertEqual(model.sc_Search(None, None)[2], 1)
        with self.assertRaises(RuntimeError):
            model.addNodes(w, u)

    def test_does_not_import_pp3(self):
        self.assertFalse(any(name == 'pp3' or name.startswith('pp3.') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
