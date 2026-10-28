"""Shared training and per-epoch accounting for the four Texture baselines."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from energy import TotalEnergyCounter
from models import Classifier
from data import add_data_arguments, load_data, data_metadata
from completion import skip_completed
from metrics import compact_metrics
from bp_memory import SavedTensorMemory, BP_MEMORY_VERSION, BP_MEMORY_FIELD
from state_memory import make_optimizer, optimizer_state_bytes, AUX_MEMORY_VERSION, SynchronizedSegment


def allocate_block_energy(rows, joules, total_energy, total_seconds):
    """Allocate one measured block by epoch time, preserving its energy sum."""
    duration = sum(row['epoch_seconds'] for row in rows)
    if duration <= 0:
        raise ValueError('Nonpositive training block duration')
    assigned = 0.
    for index, row in enumerate(rows):
        seconds = row['epoch_seconds']
        energy = None
        if joules is not None:
            energy = (joules - assigned if index == len(rows) - 1
                      else joules * seconds / duration)
            assigned += energy
        total_seconds += seconds
        total_energy = (total_energy + energy
                        if total_energy is not None and energy is not None else None)
        row.update(energy_j=energy, power_w=energy / seconds if energy is not None else None,
                   total_energy_j=total_energy,
                   total_power_w=total_energy / total_seconds if total_energy is not None else None,
                   total_seconds=total_seconds)
    return total_energy, total_seconds


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
    parser.add_argument('--eval-each-epoch', action='store_true',
                        help='Restore train/test reporting evaluation; measure energy one epoch at a time')
    parser.add_argument('--smoke', action='store_true', help='One epoch; first 16 steps, <=8 samples per source split')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    train, test = load_data(args, args.smoke)
    config = dict(defaults, epochs=1 if args.smoke else args.epochs,
                  batch_size=args.batch_size, seed=args.seed, method=method,
                  smoke=args.smoke, data_root=str(args.data_root.resolve()), device=args.device,
                  eval_each_epoch=args.eval_each_epoch,
                  energy_window_epochs=1 if args.eval_each_epoch else (10 if method == 'ann' else 5))
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
    optimizer = make_optimizer(model.parameters(), config)
    destination = args.output / args.dataset / ('smoke' if args.smoke else 'full') / method / f'seed_{args.seed}'
    config.update(train_samples=len(train), test_samples=len(test), torch_version=torch.__version__,
                  optimizer_defaults=optimizer.defaults, steps=train.steps,
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  deterministic_algorithms=True, cudnn_benchmark=False,
                  cudnn_deterministic=True,
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  energy_measurement_version='nvml_total_train_block_time_allocation_v1',
                  bp_memory_version=BP_MEMORY_VERSION,
                  auxiliary_memory_version=AUX_MEMORY_VERSION,
                  optimizer_state_measurement='epoch_end_after_last_step',
                  bp_memory_aggregation='maximum_live_saved_nonparameter_storage_per_epoch',
                  measurement_window='consecutive_training_epochs_excludes_reporting_evaluation')
    if skip_completed(destination, config):
        return
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    total_energy, total_seconds = 0.0, 0.0
    counter = TotalEnergyCounter(device)
    try:
        with (destination / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            window_epochs = config['energy_window_epochs']
            for first_epoch in range(1, config['epochs'] + 1, window_epochs):
                last_epoch = min(first_epoch + window_epochs - 1, config['epochs'])
                rows = []
                # No reporting evaluation or CSV/log writes inside this block.
                _, initial_energy = counter.read()
                for epoch in range(first_epoch, last_epoch + 1):
                    para_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    with SynchronizedSegment(device):
                        model.train()
                        loss_sum = count = 0
                        bp_memory_bytes = 0
                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            with SavedTensorMemory(model.parameters(), device) as bp_memory:
                                logits = model(x)
                                loss = F.cross_entropy(logits, y)
                                if not torch.isfinite(loss):
                                    raise FloatingPointError('Nonfinite training loss')
                                loss.backward()
                            bp_memory_bytes = max(bp_memory_bytes, bp_memory.peak_bytes)
                            optimizer.step()
                            loss_sum += loss.item() * len(y)
                            count += len(y)
                    seconds = time.perf_counter() - start
                    optimizer_bytes = optimizer_state_bytes(optimizer)
                    rows.append(compact_metrics(dict(epoch=epoch,
                        para_mem_bytes=para_mem_bytes,
                        optimizer_state_bytes=optimizer_bytes,
                        auxiliary_mem_bytes=optimizer_bytes + bp_memory_bytes,
                        **{BP_MEMORY_FIELD: bp_memory_bytes},
                        train_cross_entropy=loss_sum / count,
                        energy_j=None, power_w=None, total_energy_j=None, total_power_w=None,
                        epoch_seconds=seconds, total_seconds=None)))
                _, final_energy = counter.read()
                joules = ((final_energy - initial_energy) / 1000
                          if initial_energy is not None and final_energy is not None else None)
                total_energy, total_seconds = allocate_block_energy(
                    rows, joules, total_energy, total_seconds)
                # Disabled by default: these calls must stay outside a counter window.
                # Enabling them uses single-epoch windows, preserving each model snapshot.
                if args.eval_each_epoch:
                    train_metrics = evaluate(model, train_eval, device)
                    test_metrics = evaluate(model, test_loader, device)
                    rows[-1].update(mse=train_metrics['mse'], test_mse=test_metrics['mse'],
                                    train_acc=train_metrics['acc'], test_acc=test_metrics['acc'])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                for row in rows:
                    print(json.dumps(row), flush=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        completed_epochs=config['epochs'], config=config), destination / 'model.pt')
        (destination / 'status.json').write_text(json.dumps(dict(status='success', epochs=config['epochs'])))
    except BaseException:
        (destination / 'status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc())))
        raise
    finally:
        counter.close()
