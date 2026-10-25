"""Conservative completed-task discovery without importing torch."""
import csv
import hashlib
import math
import json
from pathlib import Path


def identity(config):
    ignored = {'output', 'entrypoint', 'device', 'gpu_token', 'experiment_id'}
    payload = {k:v for k,v in config.items() if k not in ignored}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_digest(root, script):
    # Dataset/method implementations and measurement code affect reuse validity.
    directory = Path(script).relative_to(root).parts[0]
    files = list((root/directory).rglob('*.py')) + list((root/directory).rglob('*.json'))
    files += [root/'pp3'/name for name in
              ['data.py','splits.py','native_runtime.py','flops.py','numpy_linalg.py','energy.py']]
    if directory != 'gate_zo_bdnn':
        files += list((root/'gate_zo_bdnn').glob('*.py'))
    h = hashlib.sha256()
    for path in sorted(set(files)):
        if any(part in {'__pycache__','outputs','data'} for part in path.relative_to(root).parts):
            continue
        h.update(str(path.relative_to(root)).replace('\\','/').encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def data_signature(config):
    spec=config['data']; root=Path(spec['root'])
    names=['shd_train.h5','shd_test.h5'] if config['dataset']=='shd' else [spec[k] for k in ['train_x','train_y','test_x','test_y']]
    result=[]
    for name in names:
        path=root/name
        stat=path.stat() if path.is_file() else None
        result.append(dict(path=str(path),size=stat.st_size if stat else None,
                           mtime_ns=stat.st_mtime_ns if stat else None))
    return result


def training_finished(status):
    """FLOPs coverage is diagnostic only, including legacy status labels."""
    return (bool(status.get('training_completed'))
            and status.get('status') in {'success', 'flops_incomplete'})


def completed_status(directory):
    """Check training artifacts without requiring FLOPs coverage diagnostics."""
    directory=Path(directory)
    try:
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        if not training_finished(status):
            return None
        if (directory/'model.pt').stat().st_size != status['model_bytes'] or status['model_bytes']<=0:
            return None
        with (directory/'metrics.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        if not rows or len(rows)!=status['epochs']:
            return None
        if status.get('method') == 'efficienttrain' or rows[0].get('method') == 'efficienttrain':
            # Old fixed-size ResNet runs must not be reused or summarized as ET.
            curriculum=json.loads((directory/'et_curriculum.json').read_text(encoding='utf-8'))
            protocol='efficienttrain_pp_resnet50_v1'
            if curriculum.get('protocol') != protocol or curriculum.get('actual_epochs') != len(rows):
                return None
            stages=curriculum.get('stages', [])
            expected=[stage for stage in stages for _ in range(stage['epochs'])]
            if len(stages)!=10 or len(expected)!=len(rows):
                return None
            for row, stage in zip(rows, expected):
                if (row.get('et_protocol') != protocol or int(row['et_stage']) != stage['stage']
                    or int(row['et_input_size']) != stage['input_size']
                    or int(row['et_observed_input_size']) != stage['input_size']
                    or row['et_aa'] != stage['aa']):
                    return None
        if status.get('artifact_version') in (2, 3):
            if not status.get('experiment_id'):
                return None
            epochs = [int(row['epoch']) for row in rows]
            if len(set(epochs)) != len(epochs):
                return None
            for row in rows:
                counted = float(row['epoch_flops' if status['artifact_version'] == 3 else 'epoch_known_flops'])
                if not math.isfinite(counted) or counted < 0:
                    return None
                if int(row['final_model_bytes']) != status['model_bytes']:
                    return None
            return status
        for row in rows:
            epoch=int(row['epoch'])
            if row.get('final_model_bytes') and int(row['final_model_bytes'])!=status['model_bytes']:
                return None
        return status
    except (OSError,ValueError,KeyError,TypeError):
        return None


def completed_index(root):
    result={}
    for path in sorted(Path(root).rglob('status.json')):
        if '_history' in path.relative_to(root).parts:
            continue
        status = completed_status(path.parent)
        if status and status.get('artifact_version') in (2, 3) and status.get('experiment_id'):
            result[status['experiment_id']] = path.parent
    for path in sorted(Path(root).rglob('launch.json')):
        if '_history' in path.relative_to(root).parts:
            continue
        try:
            config=json.loads(path.read_text(encoding='utf-8'))
            key=config.get('experiment_id')
            if key and identity(config)==key and completed_status(path.parent):
                result[key]=path.parent
        except (OSError,ValueError,TypeError):
            continue
    return result
