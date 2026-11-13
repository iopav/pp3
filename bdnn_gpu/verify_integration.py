"""Synthetic data only: exercise the unchanged PP3 controller and summarizer on CUDA."""
import json
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
import h5py
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
    # Synthetic events exercise the real SHD binning and deterministic training
    # augmentation. Events cover the selected smoke bins; these are not SHD data.
    spec = DATASETS['shd']
    folder = data / 'SHD'
    folder.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, spec['steps']-1, 16, dtype=int)
    times = np.repeat((bins + .25) * spec['dt'], 700).astype('float32')
    units = np.tile(np.arange(700), len(bins))
    for split in ('train', 'test'):
        with h5py.File(folder/f'shd_{split}.h5', 'w') as stream:
            stream.create_dataset('spikes/times', data=np.tile(times, (spec['classes'], 1)))
            stream.create_dataset('spikes/units', data=np.tile(units, (spec['classes'], 1)))
            stream.create_dataset('labels', data=np.arange(spec['classes']))
            stream.create_dataset('extra/speaker', data=np.arange(spec['classes']) % 4)
    command = [sys.executable, str(root/'run_experiments_bdnn_gpu.py'), '--methods', 'bdnn',
        '--datasets', 'spike', 'texture', 'shd', '--seeds', '57', '--smoke', '--gpus', '0',
        '--data-root', str(data), '--output', str(work/'results')]
    subprocess.run(command, check=True, cwd=root)
    subprocess.run([sys.executable, str(root/'summarize_experiments.py'), '--inputs',
        str(work/'results'), '--output', str(work/'summary.csv')], check=True, cwd=root)
    outputs = [work/'results'/name/'bdnn'/'seed_57' for name in ('spike', 'texture', 'shd')]
    statuses = [json.loads((p/'status.json').read_text(encoding='utf-8')) for p in outputs]
    assert len(statuses) == 3
    for status in statuses:
        assert status['training_completed'] and status['flops_complete'], status
        assert status['actual_device'] == 'cuda:0', status
        # Preserve the native one-round final truncation to zero nodes.
        assert status['final_slice_index'] == 0 and status['parameter_bytes'] == 0, status
    for output in outputs:
        path = output/'metrics.csv'
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                assert float(row['train_flops']) > 0
                assert float(row['epoch_flops']) == float(row['train_flops']) + float(row['test_flops'])
                assert float(row['train_flops']) == sum(float(v) for k, v in row.items()
                    if k.startswith('train_stage_') and k.endswith('_flops'))
                assert int(row['para_mem_bytes']) > 0
                assert int(row['auxiliary_mem_bytes']) == 0
                assert int(row['train_peak_cuda_allocated_bytes']) > 0
        for phase in output.glob('epoch_*_operations.json'):
            record = json.loads(phase.read_text(encoding='utf-8'))
            assert record['complete'] and not record['unknown'], record
            assert record['known_flops'] == sum(r['flops'] for r in record['operations'])
            assert record['known_flops'] == sum(record['known_flops_by_stage'].values())
    with (work/'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 6, summary  # One epoch plus one final_refit per dataset.
    (work/'verification.json').write_text(json.dumps(dict(synthetic_only=True,
        command=command, statuses=statuses), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
