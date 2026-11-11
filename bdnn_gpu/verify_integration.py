"""Synthetic data only: exercise the unchanged PP3 controller and summarizer on CUDA."""
import json
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
from constants import DATASETS


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / 'outputs' / 'bdnn_gpu_verification'
    data = work / 'synthetic_data'
    rng = np.random.RandomState(57)
    for name in ('spike', 'texture'):
        spec = DATASETS[name]
        folder = data / name
        folder.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'test'):
            x = rng.uniform(1, 4, (spec['classes'], spec['steps'], spec['inputs'])).astype('float32')
            np.save(folder / spec[split+'_x'], x)
            np.save(folder / spec[split+'_y'], np.arange(spec['classes'], dtype='int64'))
    command = [sys.executable, str(root/'run_experiments_bdnn_gpu.py'), '--methods', 'bdnn',
        '--datasets', 'spike', 'texture', '--seeds', '57', '--smoke', '--gpus', '0',
        '--data-root', str(data), '--output', str(work/'results')]
    subprocess.run(command, check=True, cwd=root)
    subprocess.run([sys.executable, str(root/'summarize_experiments.py'), '--inputs',
        str(work/'results'), '--output', str(work/'summary.csv')], check=True, cwd=root)
    statuses = [json.loads(p.read_text(encoding='utf-8')) for p in (work/'results').rglob('status.json')]
    assert len(statuses) == 2
    for status in statuses:
        assert status['training_completed'] and status['flops_complete'], status
        assert status['actual_device'] == 'cuda:0', status
        # Preserve the native one-round final truncation to zero nodes.
        assert status['final_slice_index'] == 0 and status['parameter_bytes'] == 0, status
    for path in (work/'results').rglob('metrics.csv'):
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                assert float(row['train_flops']) > 0
                assert float(row['epoch_flops']) == float(row['train_flops']) + float(row['test_flops'])
                assert float(row['train_flops']) == sum(float(v) for k, v in row.items()
                    if k.startswith('train_stage_') and k.endswith('_flops'))
    (work/'verification.json').write_text(json.dumps(dict(synthetic_only=True,
        command=command, statuses=statuses), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
