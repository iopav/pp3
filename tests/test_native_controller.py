import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiments


class ControllerTests(unittest.TestCase):
    def test_gpu_queue_failure_continue_and_per_seed_results(self):
        live, launched, peak = set(), [], []
        class Process:
            def __init__(self, command, cwd, env, stdout, stderr):
                self.gpu = env['CUDA_VISIBLE_DEVICES']
                self_test.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                self.config = json.loads(Path(env['PP3_CONFIG']).read_text(encoding='utf-8'))
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


if __name__ == '__main__':
    unittest.main()
