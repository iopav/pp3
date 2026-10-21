"""Shared training and per-epoch accounting for the four Texture baselines."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from energy import AuxiliaryMemory, Energy
from models import Classifier
from data import add_data_arguments, load_data, data_metadata


def evaluate(model, loader, device):
    model.eval()
    correct = count = 0
    squared = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite evaluation output')
            squared += F.mse_loss(logits.softmax(-1), F.one_hot(y, logits.shape[-1]).float(),
                                  reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += len(y)
    return dict(acc=correct / count, mse=squared / (count * logits.shape[-1]))


def main(method):
    defaults = json.loads(Path(__file__).with_name('defaults.json').read_text())
    parser = argparse.ArgumentParser(description=f'{method}: one hidden layer, 49 units, configurable sequence dataset')
    add_data_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'])
    parser.add_argument('--batch-size', type=int, default=defaults['batch_size'])
    parser.add_argument('--seed', type=int, default=defaults['seed'])
    parser.add_argument('--smoke', action='store_true', help='One epoch; first 16 steps, <=8 samples per source split')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    train, test = load_data(args, args.smoke)
    config = dict(defaults, epochs=1 if args.smoke else args.epochs,
                  batch_size=args.batch_size, seed=args.seed, method=method,
                  smoke=args.smoke, data_root=str(args.data_root.resolve()), device=args.device)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use run_all.py --cpu for an explicit CPU run')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    config.update(data_metadata(args, train))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    train_eval = DataLoader(train, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model = Classifier(method, config, inputs=train.inputs, classes=train.classes).to(device)
    optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}
    if config['optimizer'] not in optimizers:
        raise ValueError('Supported optimizers: Adam, SGD')
    optimizer = optimizers[config['optimizer']](model.parameters(), lr=config['lr'])
    destination = args.output / args.dataset / ('smoke' if args.smoke else 'full') / method / f'seed_{args.seed}'
    destination.mkdir(parents=True, exist_ok=False)
    config.update(train_samples=len(train), test_samples=len(test), torch_version=torch.__version__,
                  optimizer_defaults=optimizer.defaults, steps=train.steps,
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  deterministic_algorithms=True, cudnn_benchmark=False,
                  cudnn_deterministic=True,
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  measurement_window='optimization_excludes_train_and_test_reporting_evaluation')
    (destination / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    total_energy, total_seconds = 0.0, 0.0
    try:
        with (destination / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            for epoch in range(1, config['epochs'] + 1):
                para_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                parameter_device_bytes = sum(
                    p.numel() * p.element_size()
                    for p in model.parameters() if p.device == device
                )
                with Energy(device, config['power_sample_seconds']) as meter, \
                     AuxiliaryMemory(device, parameter_device_bytes) as memory:
                    model.train()
                    loss_sum = count = 0
                    recurrent_state_mem_bytes = 0
                    hidden_activation_mem_bytes = 0
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(x)
                        recurrent_state_mem_bytes = max(
                            recurrent_state_mem_bytes, model.recurrent_state_bytes
                        )
                        hidden_activation_mem_bytes = max(
                            hidden_activation_mem_bytes, model.hidden_activation_bytes
                        )
                        loss = F.cross_entropy(logits, y)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item() * len(y)
                        count += len(y)
                energy = meter.result()
                memory_metrics = memory.result()
                hidden_state_mem_bytes = (
                    recurrent_state_mem_bytes + hidden_activation_mem_bytes
                )
                if (memory_metrics['auxiliary_mem_bytes'] is not None
                        and memory_metrics['auxiliary_mem_bytes'] < hidden_state_mem_bytes):
                    raise RuntimeError('Measured auxiliary memory is smaller than hidden working storage')
                # Both reporting evaluations are outside the measurement contexts.
                train_metrics = evaluate(model, train_eval, device)
                test_metrics = evaluate(model, test_loader, device)
                joules, seconds = energy['gpu_energy_j'], energy['seconds']
                total_seconds += seconds
                total_energy = total_energy + joules if total_energy is not None and joules is not None else None
                row = dict(epoch=epoch,
                    para_mem_bytes=para_mem_bytes,
                    auxiliary_mem_bytes=memory_metrics['auxiliary_mem_bytes'],
                    hidden_state_mem_bytes=hidden_state_mem_bytes,
                    auxiliary_mem_method=memory_metrics['auxiliary_mem_method'],
                    auxiliary_mem_reason=memory_metrics['auxiliary_mem_reason'],
                    mse=train_metrics['mse'], test_mse=test_metrics['mse'],
                    train_acc=train_metrics['acc'], test_acc=test_metrics['acc'],
                    train_cross_entropy=loss_sum / count,
                    energy_j=joules, power_w=joules / seconds if joules is not None else None,
                    total_energy_j=total_energy,
                    total_power_w=total_energy / total_seconds if total_energy is not None else None,
                    epoch_seconds=seconds, total_seconds=total_seconds,
                    energy_method=energy['energy_method'], energy_reason=energy['energy_reason'],
                    gpu_uuid=energy['gpu_uuid'])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
        torch.save(dict(model=model.state_dict(), config=config), destination / 'model.pt')
        (destination / 'status.json').write_text(json.dumps(dict(status='success', epochs=config['epochs'])))
    except BaseException:
        (destination / 'status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc())))
        raise
