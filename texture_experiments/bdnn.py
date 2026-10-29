"""Fixed-width ordinary BDNN using the unmodified BDNNversion1-main primitives.

The native implementation computes on CPU, even when a GPU is selected for
telemetry. GPU board energy is not CPU training energy.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import importlib.util

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
from data import add_data_arguments, load_data, data_metadata
from completion import skip_completed
from energy import TotalEnergyCounter
from train import allocate_block_energy
from startup_power import report_startup_power
from bp_memory import BP_MEMORY_FIELD

SOURCE = Path(__file__).resolve().parents[1] / 'BDNNversion1-main'
sys.path.insert(0, str(SOURCE))
def source_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bdnn1 = source_module('_texture_ordinary_bdnn', 'bdnn1.py').bdnn1
SCN_DEFAULTS = source_module('_texture_ordinary_bdnn_config', 'config.py').SCN_DEFAULTS
sys.path.remove(str(SOURCE))


def finite(value):
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError('Nonfinite BDNN state/output')


def width(model):
    return 0 if isinstance(model.W, list) else model.W.shape[1]


def grow_one(model, x, targets):
    # Native classification searches against T throughout; preserve this rule.
    w, u, failed = model.sc_Search(x, targets)
    if failed:
        raise RuntimeError(f'BDNN candidate search failed at hidden={width(model)}')
    finite(w)
    finite(u)
    before = width(model)
    model.addNodes(w, u)
    if width(model) != before + 1:
        raise RuntimeError('BDNN must add exactly one hidden neuron')
    _, loss = model.upgradeSCN(x, targets)
    for value in (model.W, model.U, model.Beta, loss):
        finite(value)
    output = model.getOutput(x)
    finite(output)
    return float((output.argmax(1) == targets.argmax(1)).mean())


def run_growth(model, x, targets, target_width, counter, output, test_evaluation):
    rows = []
    total_energy = total_seconds = 0.0

    def row(seconds, accuracy):
        return dict(epoch=int(model.L), hidden_neurons=width(model),
                    para_mem_bytes=sum(np.asarray(v).nbytes for v in (model.W, model.U, model.Beta)),
                    optimizer_state_bytes=0, auxiliary_mem_bytes=0, **{BP_MEMORY_FIELD: 0},
                    train_acc=accuracy, final_test_acc=None, energy_j=None, power_w=None,
                    total_energy_j=None, total_power_w=None, epoch_seconds=seconds, total_seconds=None)

    def flush(block, initial):
        nonlocal total_energy, total_seconds
        _, final = counter.read()
        joules = (final - initial) / 1000 if initial is not None and final is not None else None
        total_energy, total_seconds = allocate_block_energy(block, joules, total_energy, total_seconds)
        rows.extend(block)

    def save():
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    while width(model) < target_width:
        _, initial = counter.read()
        block = []
        try:
            for _ in range(min(5, target_width - width(model))):
                started = time.perf_counter()
                acc = grow_one(model, x, targets)
                block.append(row(time.perf_counter() - started, acc))
        except BaseException:
            block.append(row(time.perf_counter() - started, None))
            flush(block, initial)
            save()
            raise
        flush(block, initial)
        save()
        for item in block:
            print(json.dumps(item), flush=True)
    _, initial = counter.read()
    started = time.perf_counter()
    h, _ = model.getH(x)
    model.computeBeta(h, targets)
    finite(model.Beta)
    flush([row(time.perf_counter() - started, None)], initial)
    prediction = model.getOutput(x)
    finite(prediction)
    rows[-1]['train_acc'] = float((prediction.argmax(1) == targets.argmax(1)).mean())
    rows[-1]['final_test_acc'] = test_evaluation(model)
    save()
    (output / 'final_metrics.json').write_text(json.dumps(rows[-1], indent=2), encoding='utf-8')
    print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--device', default='cuda:0', help='Telemetry device; native BDNN computes on CPU')
    parser.add_argument('--epochs', type=int, default=100, help='Compatibility only: BDNN grows to max-hidden')
    parser.add_argument('--batch-size', type=int, default=128, help='Compatibility only: native full-array computation')
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--max-hidden', type=int, default=49)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.max_hidden, args.epochs, args.batch_size) < 1:
        parser.error('max-hidden, epochs and batch-size must be positive')
    report_startup_power(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train, test = load_data(args)

    def arrays(dataset):
        pairs = [dataset[i] for i in range(len(dataset))]
        if args.smoke:
            pairs = [p for label in range(train.classes) for p in [p for p in pairs if p[1] == label][:4]]
        x, y = zip(*pairs)
        x = torch.stack(x).numpy()
        return (x[:, :16] if args.smoke else x), np.eye(train.classes, dtype=np.float32)[list(y)]

    x, targets = arrays(train)
    x_test, test_targets = arrays(test)
    native = dict(SCN_DEFAULTS, nB=1, L_max=args.max_hidden)
    config = dict(method='bdnn', max_hidden=args.max_hidden, layers=1, native_args=native,
                  seed=args.seed, source=str(SOURCE), smoke=args.smoke,
                  data_root=str(args.data_root.resolve()), compute_device='cpu', telemetry_device=args.device,
                  epochs_unit='growth_attempt', energy_window_epochs=5,
                  energy_measurement_version='nvml_total_growth_block_time_allocation_v1',
                  accuracy_reporting_version='growth_train_final_test_v1',
                  stopping_policy='fixed_width_candidate_failure_is_failure',
                  train_samples=len(x), test_samples=len(x_test))
    config.update(data_metadata(args, train))
    config['steps'] = x.shape[1]
    destination = args.output / args.dataset / ('smoke' if args.smoke else 'full') / 'bdnn' / f'seed_{args.seed}'
    if skip_completed(destination, config):
        return
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    counter = TotalEnergyCounter(torch.device(args.device))
    try:
        model = bdnn1(**native)
        def final_test(model):
            prediction = model.getOutput(x_test)
            finite(prediction)
            return float((prediction.argmax(1) == test_targets.argmax(1)).mean())
        run_growth(model, x, targets, args.max_hidden, counter, destination, final_test)
        if width(model) != args.max_hidden:
            raise RuntimeError('BDNN did not reach target width')
        torch.save(dict(W=torch.as_tensor(model.W), U=torch.as_tensor(model.U), Beta=torch.as_tensor(model.Beta),
                        hidden_neurons=width(model), growth_round=int(model.L), config=config), destination / 'model.pt')
        (destination / 'summary.json').write_text(json.dumps(dict(hidden_neurons=width(model), compute_device='cpu')))
        (destination / 'status.json').write_text(json.dumps(dict(status='success', hidden_neurons=width(model))))
    except BaseException:
        (destination / 'status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc())))
        raise
    finally:
        counter.close()


if __name__ == '__main__':
    main()
