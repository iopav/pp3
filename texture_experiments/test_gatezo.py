"""Regression checks for the actual-neuron limit and runtime isolation."""
import sys
import inspect
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from gatezo import BDNN1GateZO, BoundedGateZO, native_settings
from gate_zo_bdnn.structural_gate import run_gatezo_growth


class GateZOTests(unittest.TestCase):
    def test_native_defaults_and_launcher_routing(self):
        settings = native_settings(device='cpu')
        self.assertEqual((settings['nB'], settings['L_max']), (1, 49))
        model = BoundedGateZO(**settings)
        self.assertEqual((model.nB, model.L_max), (1, 49))
        import run_all
        with (patch.object(sys, 'argv', ['run_all', '--cpu', '--epochs', '100']),
              contextlib.redirect_stdout(io.StringIO()),
              patch.object(run_all.subprocess, 'run') as run):
            run.return_value.returncode = 0
            self.assertEqual(run_all.main(), 0)
        self.assertEqual(run.call_count, 5)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('--epochs') + 1], '100')
            self.assertNotIn('--nB', command)
            self.assertNotIn('--L-max', command)

    def test_candidate_failure_stops_without_retry(self):
        model = BoundedGateZO(**native_settings(device='cpu'))
        x = np.ones((20, 1, 9), dtype=np.float32)
        targets = np.eye(2, dtype=np.float32)[np.arange(20) % 2]
        with patch.object(BDNN1GateZO, 'sc_Search', return_value=([], [], 1)) as search:
            with self.assertRaises(RuntimeError):
                model.classification(x, targets, x, targets)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(model.L, 0)

    def test_native_round_limit_is_enforced_before_neuron_target(self):
        model = BoundedGateZO(max_hidden=3, **native_settings(rounds=2, device='cpu'))
        x = np.ones((20, 1, 9), dtype=np.float32)
        targets = np.eye(2, dtype=np.float32)[np.arange(20) % 2]
        with (patch.object(BDNN1GateZO, 'sc_Search',
                           return_value=(np.ones((9, 1)), np.zeros((1, 1)), 0)) as search,
              patch.object(BoundedGateZO, 'getH',
                           side_effect=lambda data: (np.ones((len(data), model.hidden_count)), None)),
              patch.object(model.structural_gate, 'select', return_value={'kept': 1})):
            with self.assertRaises(RuntimeError):
                model.classification(x, targets, x, targets)
        self.assertEqual(search.call_count, 2)
        self.assertEqual(model.L, 2)

    def test_49_single_neuron_rounds_and_target_checkpoint(self):
        original_source = inspect.getsource(run_gatezo_growth)
        original_code = run_gatezo_growth.__code__
        model = BoundedGateZO(max_hidden=49, compute_device='cpu', nB=1,
                              L_max=49, tol=1e9, verbose=1,
                              growth_protocol='validation_checkpoint', gate_zo_seed=22,
                              gate_zo_verbose=False)
        x = np.ones((20, 1, 9), dtype=np.float32)
        targets = np.eye(2, dtype=np.float32)[np.arange(20) % 2]
        calls = []
        def proposal(instance, x, targets):
            calls.append(instance.nB)
            return np.ones((9, instance.nB)), np.zeros((instance.nB, instance.nB)), 0
        def hidden(instance, data):
            return np.ones((len(data), model.hidden_count), dtype=np.float32), None
        # Flat accuracy and tiny error would stop the native loop, while its
        # historical best checkpoint would have only one neuron.
        with patch.object(BDNN1GateZO, 'sc_Search', proposal), \
             patch.object(BoundedGateZO, 'getH', hidden), \
             patch.object(model.structural_gate, 'select',
                          side_effect=lambda m, x, t, old, proposed: {'kept': proposed}):
            model.classification(x, targets, x, targets)
        self.assertEqual(calls, [1] * 49)
        self.assertEqual(model.L, 49)
        self.assertEqual(model.hidden_count, 49)
        self.assertEqual(model.gate_zo_summary['attempted_rounds'], 49)
        self.assertEqual(model.gate_zo_summary['hidden_size'], 49)
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
