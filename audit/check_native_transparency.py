"""Compare the same original entry with and without FLOP dispatch accounting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('reference_run', type=Path)
parser.add_argument('--output', type=Path, default=ROOT/'audit/native_transparency')
args = parser.parse_args()


def digest(value):
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().contiguous().numpy()
        if isinstance(obj, np.ndarray):
            h.update(str(obj.dtype).encode())
            h.update(str(obj.shape).encode())
            h.update(obj.tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj):
                h.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


report = []
for launch in sorted(args.reference_run.glob('*/spike/seed_57/launch.json')):
    config = json.loads(launch.read_text(encoding='utf-8'))
    outputs = []
    for enabled in (True, False):
        output = args.output/config['method']/('counted' if enabled else 'uncounted')
        output.mkdir(parents=True, exist_ok=False)
        config.update(output=str(output.resolve()), count_flops=enabled)
        path = output/'launch.json'
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        env = os.environ.copy()
        env.update(PP3_CONFIG=str(path.resolve()), CUDA_VISIBLE_DEVICES='',
                   PYTHONHASHSEED=str(config['seed']))
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH')]))
        script = Path(config['entrypoint'])
        with (output/'run.log').open('w', encoding='utf-8') as log:
            code = subprocess.run([sys.executable, str(script)], cwd=script.parent,
                                  env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        assert code == 0, (output, code)
        status = json.loads((output/'status.json').read_text(encoding='utf-8'))
        assert status['training_completed'], output
        state = torch.load(output/'model.pt', map_location='cpu', weights_only=False)
        with (output/'metrics.csv').open(encoding='utf-8', newline='') as f:
            metrics = [{k: row.get(k) for k in ('epoch', 'train_accuracy', 'test_accuracy')}
                       for row in csv.DictReader(f)]
        outputs.append((digest(state), metrics))
    assert outputs[0] == outputs[1], config['method']
    report.append(dict(method=config['method'], identical=True, model_sha256=outputs[0][0]))
    print(config['method'], 'identical', flush=True)
(args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
