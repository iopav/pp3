"""Conservative completed-task discovery without importing torch."""
import csv
import hashlib
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


def completed_status(directory):
    """Check completion reports, model size and per-epoch training records."""
    directory=Path(directory)
    try:
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        if not status.get('training_completed') or status.get('status') not in {'success','flops_incomplete'}:
            return None
        if (directory/'model.pt').stat().st_size != status['model_bytes'] or status['model_bytes']<=0:
            return None
        with (directory/'metrics.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        if not rows or len(rows)!=status['epochs']:
            return None
        if status.get('artifact_version') == 2:
            if not status.get('experiment_id'):
                return None
            epochs = [int(row['epoch']) for row in rows]
            if len(set(epochs)) != len(epochs):
                return None
            for row in rows:
                if row.get('epoch_flops_complete') not in {'True', 'False'}:
                    return None
                if float(row['epoch_known_flops']) < 0:
                    return None
                if int(row['final_model_bytes']) != status['model_bytes']:
                    return None
            return status
        for row in rows:
            epoch=int(row['epoch'])
            phase=json.loads((directory/f'epoch_{epoch:04d}_train_operations.json').read_text(encoding='utf-8'))
            if 'complete' not in phase:
                return None
            if row.get('final_model_bytes') and int(row['final_model_bytes'])!=status['model_bytes']:
                return None
            if row.get('test_accuracy'):
                test=json.loads((directory/f'epoch_{epoch:04d}_test_operations.json').read_text(encoding='utf-8'))
                if 'complete' not in test:
                    return None
        for path in directory.glob('epoch_*_operations.json'):
            if 'complete' not in json.loads(path.read_text(encoding='utf-8')):
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
        if status and status.get('artifact_version') == 2 and status.get('experiment_id'):
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
