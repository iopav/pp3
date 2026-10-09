import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pp3.flops import FlopCounter

ROOT = Path(__file__).resolve().parents[1]


def load_bundle(name):
    path = ROOT / (name+'_bdnn_gatezo_bundle') / 'gate_zo_bdnn/structural_gate.py'
    spec = importlib.util.spec_from_file_location('test_structural_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GateZONumpyAccountingTests(unittest.TestCase):
    def test_both_bundles_pinv_ridge_prediction_and_no_double_count(self):
        x = np.array([[1., 0.], [0., 1.], [1., 1.]])
        y = np.arange(12., dtype=float).reshape(3, 4)
        for name in ('spike', 'texture'):
            module = load_bundle(name)
            for ridge, expected in ((0., 193), (.5, 109)):
                baseline = module.fit_readout(x, y, ridge)
                counter = FlopCounter()
                with counter, counter.section('train'):
                    result = module.fit_readout(x, y, ridge, counter)
                np.testing.assert_array_equal(result, baseline)
                self.assertEqual(counter.summary()['flops'], expected)
                with counter, counter.section('test'):
                    metrics = module.classification_metrics(x, y, result, counter)
                self.assertEqual(metrics, module.classification_metrics(x, y, baseline))
                self.assertEqual(counter.summary()['known_flops_by_stage'], {'train': expected, 'test': 48})
                self.assertFalse(any(r['operator'].startswith('aten.') for r in counter.summary()['operations']))

    def test_failed_solve_counts_fallback_once_and_retains_unknown(self):
        for name in ('spike', 'texture'):
            module = load_bundle(name)
            x, y = np.ones((3, 2)), np.ones((3, 4))
            counter = FlopCounter()
            with patch.object(np.linalg, 'solve', side_effect=np.linalg.LinAlgError('singular')) as solver, \
                 patch.object(np.linalg, 'pinv', wraps=np.linalg.pinv) as inverse, counter:
                result = module.fit_readout(x, y, .5, counter)
            self.assertEqual(solver.call_count, 1)
            self.assertEqual(inverse.call_count, 1)
            np.testing.assert_array_equal(result, np.linalg.pinv(x.T@x+.5*np.eye(2)) @ (x.T@y))
            self.assertEqual(counter.summary()['known_flops'], 223)
            self.assertFalse(counter.summary()['complete'])
            self.assertEqual([r['operator'] for r in counter.summary()['unknown']], ['numpy::solve_failed'])

    def test_empty_disabled_and_paused(self):
        module = load_bundle('spike')
        counter = FlopCounter()
        with counter:
            output = module.fit_readout(np.empty((3, 0)), np.ones((3, 2)), 1., counter)
        self.assertEqual(output.shape, (0, 2))
        self.assertEqual(counter.summary()['known_flops'], 0)
        for enabled, paused in ((False, False), (True, True)):
            counter = FlopCounter(enabled=enabled)
            counter.paused = paused
            with counter:
                module.fit_readout(np.eye(2), np.eye(2), 1., counter)
            self.assertFalse(counter.records)

    def test_native_entries_transparency_and_actual_solver_calls(self):
        from constants import make_config
        for dataset in ('spike', 'texture'):
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config = make_config('gatezo', dataset)
                config['data'].update(root=str(directory), steps=6, inputs=2, classes=2)
                config['model'].update(epochs=1, pool_size=8, growth_step=2, batch_size=4)
                config['model']['ridge'] = .5 if dataset == 'texture' else 0.
                config.update(device='cpu', threads=1)
                for split in ('train', 'test'):
                    np.save(directory/config['data'][split+'_x'], np.full((12,6,2), 3., dtype=np.float32))
                    np.save(directory/config['data'][split+'_y'], np.tile([0,1], 6))
                runs = []
                for enabled in (False, True):
                    output = directory/str(enabled)
                    config.update(output=str(output), count_flops=enabled)
                    launch = directory/(str(enabled)+'.json')
                    launch.write_text(json.dumps(config), encoding='utf-8')
                    env = os.environ.copy()
                    env.update(PP3_CONFIG=str(launch), PYTHONPATH=str(ROOT))
                    script = ROOT/(dataset+'_bdnn_gatezo_bundle')/'run_texture.py'
                    probe = (
                        "import runpy,sys,json,os; from pathlib import Path; "
                        "import numpy as np; import torch; "
                        "runpy.run_path(sys.argv[1],run_name='__main__'); "
                        "out=Path(json.loads(Path(os.environ['PP3_CONFIG']).read_text())['output']); "
                        "torch.save((np.random.get_state(),torch.random.get_rng_state()),out/'rng.pt')"
                    )
                    proc = subprocess.run([sys.executable, '-c', probe, str(script)], cwd=script.parent, env=env,
                                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
                    self.assertEqual(proc.returncode, 0, proc.stderr[-6000:])
                    state = torch.load(output/'model.pt', weights_only=False)
                    summary = json.loads((output/'native_summary.json').read_text())
                    # Native history contains timing, so compare the scientific outputs.
                    rng = torch.load(output/'rng.pt', weights_only=False)
                    runs.append((state, {k: summary[k] for k in ('training','test','hidden_size','checkpoint_accuracy')}, rng))
                    phases = [json.loads(p.read_text()) for p in output.glob('*_operations.json')]
                    if enabled:
                        rows = [r for phase in phases for r in phase['operations']]
                        solver_name = 'manual::numpy::solve' if config['model']['ridge'] else 'manual::numpy::pinv'
                        self.assertTrue(any(r['operator']==solver_name for r in rows))
                        self.assertTrue(any(r['operator']=='manual::numpy::matmul' for r in rows))
                        self.assertGreater(sum(r['calls'] for r in rows if r['operator']==solver_name), 1)
                    else:
                        self.assertTrue(all(p['known_flops']==0 for p in phases))
                    for phase in phases:
                        self.assertEqual(sum(r['flops'] for r in phase['operations']), phase['known_flops'])
                for key in runs[0][0]:
                    np.testing.assert_array_equal(np.asarray(runs[0][0][key]), np.asarray(runs[1][0][key]))
                self.assertEqual(runs[0][1], runs[1][1])
                np.testing.assert_array_equal(runs[0][2][0][1], runs[1][2][0][1])
                self.assertEqual(runs[0][2][0][2:], runs[1][2][0][2:])
                self.assertTrue(torch.equal(runs[0][2][1], runs[1][2][1]))


if __name__ == '__main__':
    unittest.main()
