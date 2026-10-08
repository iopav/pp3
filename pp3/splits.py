"""Shared outer validation indices, independent of method/model RNG."""
import hashlib
import json
from pathlib import Path
import numpy as np
import os
from contextlib import contextmanager


@contextmanager
def _lock(path):
    # OS locks are released on process exit, including a killed Slurm job.
    with path.open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream,fcntl.LOCK_UN)


def outer_split(labels, config, speakers=None):
    labels=np.asarray(labels)
    directory=config.get('split_dir')
    if directory is None and config.get('output'):
        directory=str(Path(config['output']).parent/'_splits')
    if directory is None:
        return _generate(labels,config,speakers)[:2]
    root=Path(directory)/config['dataset'];root.mkdir(parents=True,exist_ok=True)
    path=root/f'seed_{config["seed"]}.json'
    signature=dict(dataset=config['dataset'],seed=config['seed'],
        policy=config.get('validation',dict(ratio=.2,speakers=2)),
        labels_sha256=hashlib.sha256(labels.astype(np.int64).tobytes()).hexdigest(),
        speakers_sha256=hashlib.sha256(json.dumps(None if speakers is None else
            np.asarray(speakers).astype(str).tolist()).encode()).hexdigest())
    with _lock(path.with_suffix('.lock')):
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved.get('input_signature')!=signature:
                raise ValueError(f'Cached validation indices differ from data/policy: {path}; choose a new --split-dir')
            train=np.asarray(saved['train_indices'],dtype=np.int64)
            val=np.asarray(saved['validation_indices'],dtype=np.int64)
            if not np.array_equal(np.sort(np.concatenate([train,val])),np.arange(len(labels))):
                raise ValueError(f'Invalid cached validation indices: {path}')
            return train,val
        train,val,payload=_generate(labels,config,speakers)
        payload['input_signature']=signature
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        temporary.replace(path)
        return train,val


def _generate(labels, config, speakers=None):
    labels = np.asarray(labels)
    policy = config.get('validation', dict(ratio=.2, speakers=2))
    if config['dataset'] == 'shd':
        if speakers is None:
            raise ValueError('Shared SHD validation requires official extra/speaker metadata')
        from gate_zo_bdnn.train_shd_bdnn_rsnn_blocks import speaker_disjoint_split
        train, val, held = speaker_disjoint_split(speakers, policy['speakers'], config['seed'])
        rule = 'gatezo_speaker_disjoint'
    else:
        rng = np.random.RandomState(config['seed'])
        train, val = [], []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            if len(indices) < 2:
                raise ValueError(f'Class {label} needs at least two samples for train/validation')
            rng.shuffle(indices)
            count = max(1, min(len(indices)-1, int(np.ceil(len(indices)*policy['ratio']))))
            val.extend(indices[:count]); train.extend(indices[count:])
        train, val = np.sort(train), np.sort(val)
        held, rule = [], 'stratified_holdout_ceil_per_class'
    payload = dict(dataset=config['dataset'], seed=config['seed'], rule=rule,
                   validation_ratio=policy['ratio'], validation_speakers=held,
                   train_indices=train.tolist(), validation_indices=val.tolist())
    payload['split_sha256'] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return train, val, payload
