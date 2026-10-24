"""Small, dependency-free export of the four requested metrics."""
import csv
import json
from pathlib import Path


METADATA_FIELDS = ['experiment_id', 'record_type', 'output']
FIELDS = ['method', 'dataset', 'seed', 'epoch',
          'energy_j', 'mem_bytes', 'train_acc', 'test_acc', 'flops',
          'seconds', 'train_energy_j',
          'train_flops', 'test_energy_j', 'test_flops', 'readout_fit_energy_j',
          'readout_fit_flops', 'status']


def export_summary(run_directory, tasks=None, destination=None):
    """Export metrics plus row metadata; include final costs exactly once.

    CSV data rows are numbered from 1 in the adjacent .metadata.json file.
    Keep identity and phase information there without adding metric columns.
    """
    root = Path(run_directory)
    if tasks is None:
        tasks = json.loads((root/'tasks.json').read_text(encoding='utf-8'))
    rows = []
    for task in tasks:
        out = Path(task['output'])
        status_file = out/'status.json'
        status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}
        size = status.get('model_bytes')
        base = {k: task[k] for k in ('method', 'dataset', 'seed')}
        base['experiment_id']=task.get('experiment_id','')
        base.update(mem_bytes=size,
                    status=status.get('status', 'failed'), output=str(out))
        metrics = out/'metrics.csv'
        source = []
        if metrics.exists():
            with metrics.open(encoding='utf-8', newline='') as stream:
                source = list(csv.DictReader(stream))
        for record in source:
            row = dict(base, record_type='epoch', epoch=record['epoch'])
            mapping = dict(energy_j='epoch_gpu_energy_j', train_acc='train_accuracy',
                           test_acc='test_accuracy', flops='epoch_flops',
                           seconds='epoch_seconds', train_energy_j='train_gpu_energy_j',
                           train_flops='train_flops', test_energy_j='test_gpu_energy_j',
                           test_flops='test_flops', readout_fit_energy_j='readout_fit_gpu_energy_j',
                           readout_fit_flops='readout_fit_flops')
            row.update({key: record.get(value) for key, value in mapping.items()})
            if row.get('flops') in (None,''):
                row['flops']=record.get('epoch_known_flops')
            for phase in ('train','test','readout_fit'):
                if row.get(phase+'_flops') in (None,''):
                    row[phase+'_flops']=record.get(phase+'_known_flops')
            rows.append(row)
        # Preparation and final refit are not charged repeatedly to each epoch.
        one_time = status.get('one_time_metrics', [])
        if status.get('artifact_version') not in (2, 3):
            one_time = [dict(json.loads(path.read_text(encoding='utf-8')),
                        phase=path.stem.split('_', 2)[2].removesuffix('_operations'))
                        for path in sorted(out.glob('epoch_*_operations.json'))]
        for record in one_time:
            phase = record['phase']
            if phase != 'prepare' and not phase.startswith('final_'):
                continue
            row = dict(base, record_type=phase, epoch=None,
                       energy_j=record.get('gpu_energy_j'), seconds=record.get('seconds'),
                       flops=record.get('flops') if record.get('flops') is not None else record.get('known_flops'))
            if phase.startswith('final_') and 'test' in phase:
                row['test_acc'] = record.get('accuracy')
            rows.append(row)
        if not source:
            rows.append(dict(base, record_type='no_epoch_result'))
    destination = Path(destination) if destination is not None else root/'summary.csv'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)
    destination.with_suffix('.metadata.json').write_text(json.dumps(dict(
        schema_version=1,
        csv_file=destination.name,
        row_numbering='1-based data rows, excluding the CSV header',
        rows=[dict(csv_data_row=index, **{key: row.get(key) for key in METADATA_FIELDS})
              for index, row in enumerate(rows, 1)]
    ), indent=2, ensure_ascii=False), encoding='utf-8')
    return destination
