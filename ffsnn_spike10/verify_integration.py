"""Exercise real PP3 launch, training, measurement and summary with synthetic input."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from constants import DATASETS
from pp3.completed import completed_status
from pp3.reporting import FIELDS


def main():
    root = Path(__file__).resolve().parents[1]
    work = root/'outputs/ffsnn10_verification'
    data = work/'synthetic_data/spike'
    data.mkdir(parents=True, exist_ok=True)
    spec = DATASETS['spike']
    rng = np.random.RandomState(57)
    for split in ('train', 'test'):
        x = (rng.random_sample((20, 400, 64)) < .1).astype('float32')
        np.save(data/spec[split+'_x'], x)
        np.save(data/spec[split+'_y'], np.arange(20, dtype='int64'))
    command = [sys.executable, str(root/'run_experiments_ffsnn10.py'),
        '--methods', 'ffsnn10', '--datasets', 'spike', '--seeds', '57', '--smoke',
        '--workers', '0', '--gpus', '0', '--data-root', str(data.parent),
        '--output', str(work/'results')]
    subprocess.run(command, cwd=root, check=True)
    output = work/'results/spike/ffsnn10/seed_57'
    status = completed_status(output)
    assert status and status['actual_device'] == 'cuda:0', status
    assert status['flops_complete'], (output/'flops_unknown.json').read_text()
    assert status['input_transform']['model_steps'] == 10
    with (output/'metrics.csv').open(newline='', encoding='utf-8') as stream:
        row, = csv.DictReader(stream)
    assert float(row['train_flops']) > 0 and float(row['test_flops']) > 0
    assert float(row['epoch_flops']) == float(row['train_flops']) + float(row['test_flops'])
    subprocess.run([sys.executable, str(root/'summarize_experiments.py'), '--inputs',
        str(work/'results'), '--output', str(work/'summary.csv')], cwd=root, check=True)
    with (work/'summary.csv').open(encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS
        summary = list(reader)
    assert len(summary) == 2 and all(r['method'] == 'ffsnn10' for r in summary)
    (work/'verification.json').write_text(json.dumps(dict(synthetic_only=True,
        command=command, status=status, metrics=row), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
