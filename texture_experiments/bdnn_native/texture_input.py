"""Dataset and observation bridge; no model settings or random-state changes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch


def configuration():
    return json.loads(Path(os.environ['TEXTURE_BDNN_CONFIG']).read_text(encoding='utf-8'))


def load_texture_data():
    path = Path(__file__).resolve().parents[1]/'data.py'
    spec = importlib.util.spec_from_file_location('_texture_bdnn_data', path)
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    c = configuration()
    train, test = data.datasets(c['data_root'], False, c['dataset'], c['shd_dt'],
                                c['shd_max_time'], c['shd_channel_group_size'])
    values = []
    for dataset in (train, test):
        pairs = [dataset[i] for i in range(len(dataset))]
        values.extend((torch.stack([p[0] for p in pairs]),
                       torch.nn.functional.one_hot(torch.tensor([p[1] for p in pairs]), train.classes).float()))
    return tuple(values)


def make_metrics(model):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from native_bdnn_metrics import NativeMetrics
    c = configuration()
    return NativeMetrics(model, Path(os.environ['TEXTURE_BDNN_CONFIG']).parent, c['telemetry_device'])
