import contextlib
import csv
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiments


class ControllerTests(unittest.TestCase):
    def test_real_child_traceback_and_missing_status_are_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(method='tp', dataset='spike', seed=57, output=directory)
            path = Path(directory)/'run.log'
            with path.open('w', encoding='utf-8') as log:
                child = subprocess.run([sys.executable, '-c',
                    'raise RuntimeError("training exploded")'], stdout=log, stderr=subprocess.STDOUT)
                code, _ = run_experiments.record_exit(config, log, child.returncode)
            text = path.read_text(encoding='utf-8')
            self.assertNotEqual(code, 0)
            self.assertIn('RuntimeError: training exploded', text)
            self.assertIn('status=failed', text)
            self.assertIn('process_exit_code=1', text)

    def test_final_status_diagnostics(self):
        cases = [
            (0, {'status':'flops_incomplete', 'training_completed':True,
                 'flops_missing':['aten.unknown: unknown']}, None, 'FLOPs coverage incomplete (diagnostic only); see flops_unknown.json', 0),
            (0, {'status':'flops_incomplete', 'training_completed':False}, None, 'training_completed=False', 2),
            (0, [], None, 'Invalid status report', 2),
            (1, {'status':'success', 'training_completed':True}, None, 'status=failed', 1),
            (-9, {}, None, 'Child terminated by signal SIGKILL' if os.name != 'nt' else 'Child exited with code -9', -9),
            (0, {}, 'Controller received signal SIGTERM', 'Controller received signal SIGTERM', 2),
        ]
        for raw, status, interruption, expected, effective in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                config = dict(method='tp', dataset='spike', seed=57, output=directory)
                path = Path(directory)
                (path/'status.json').write_text(json.dumps(status), encoding='utf-8')
                with (path/'run.log').open('w', encoding='utf-8') as log:
                    code, _ = run_experiments.record_exit(config, log, raw, interruption)
                self.assertEqual(code, effective)
                self.assertIn(expected, (path/'run.log').read_text(encoding='utf-8'))
                self.assertEqual(json.loads((path/'status.json').read_text())['process_exit_code'], raw)

    def test_controller_signal_stops_child_and_logs_reason(self):
        class Process:
            def __init__(self, *args, **kwargs):
                self.stopped = False
            def poll(self):
                return -15 if self.stopped else None
            def terminate(self):
                self.stopped = True
            def wait(self, timeout=None):
                return -15
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            argv = ['run_experiments.py', '--methods', 'tp', '--datasets', 'spike',
                    '--seeds', '57', '--cpu', '--output', directory]
            with patch('sys.argv', argv), patch.object(run_experiments.subprocess, 'Popen', Process), \
                 patch.object(run_experiments.time, 'sleep',
                              side_effect=lambda _: run_experiments.controller_signal(signal.SIGTERM, None)), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                run_experiments.main()
            text = (Path(directory)/'spike/tp/seed_57/run.log').read_text(encoding='utf-8')
            self.assertIn('Controller received signal SIGTERM', text)
            self.assertIn('status=failed', text)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_gpu_queue_failure_continue_and_per_seed_results(self):
        live, launched, peak = set(), [], []
        class Process:
            def __init__(self, command, cwd, env, stdout, stderr):
                self.gpu = env['CUDA_VISIBLE_DEVICES']
                self_test.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                self.config = json.loads(env['PP3_CONFIG_JSON'])
                self.output = Path(self.config['output'])
                self.code = 1 if not launched else 0
                self.polls, self.finished = 0, False
                self_test.assertNotIn(self.gpu, live)
                self_test.assertEqual(Path(command[1]), run_experiments.entrypoint(self.config))
                if len(launched)==2:
                    # GPU 0 must take its next task while GPU 1 is still busy.
                    self_test.assertIn('1',live)
                live.add(self.gpu)
                peak.append(len(live))
                launched.append(self.config)
            def poll(self):
                self.polls += 1
                if self.polls < (7 if self.gpu=='1' else 3):
                    return None
                if not self.finished:
                    self.finished = True
                    live.remove(self.gpu)
                    (self.output/'status.json').write_text(json.dumps(dict(
                        status='failed' if self.code else 'success')), encoding='utf-8')
                    with (self.output/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                        writer = csv.DictWriter(stream, fieldnames=['method','dataset','seed','epoch'])
                        writer.writeheader()
                        writer.writerow({key: self.config[key] if key != 'epoch' else 1 for key in writer.fieldnames})
                return self.code
        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            argv = ['run_experiments.py','--methods','spsn','tp','--datasets','spike',
                    '--seeds','57','58','--gpus','0','1','--output',directory]
            with patch.dict(os.environ, {}, clear=True), patch('sys.argv', argv), \
                 patch.object(run_experiments.subprocess, 'Popen', Process), \
                 patch.object(run_experiments.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                code = run_experiments.main()
            self.assertEqual(code, 1)
            self.assertEqual(len(launched), 4)
            self.assertEqual(max(peak), 2)
            self.assertFalse(live)
            paths=list(Path(directory).glob('spike/*/seed_*/metrics.csv'))
            self.assertEqual(len(paths),4)
            statuses=[json.loads(p.with_name('status.json').read_text()) for p in paths]
            self.assertEqual(sum(s['status']=='success' for s in statuses),3)
            self.assertFalse((Path(directory)/'_runs').exists())
            self.assertFalse((Path(directory)/'_history').exists())
            self.assertFalse(list(Path(directory).rglob('launch.json')))


if __name__ == '__main__':
    unittest.main()
