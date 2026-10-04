"""Checked one-time additions to existing bundle entries and measurement hooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for dataset in ('spike', 'texture'):
    package = ROOT/f'{dataset}_bdnn_gatezo_bundle/gate_zo_bdnn'
    entry = package/'run_texture_comparison_task.py'
    source = entry.read_text(encoding='utf-8')
    assert 'PP3_CONFIG' not in source
    source = source.replace('def main():\n    args = parse_args()', '''def main():
    import os
    if os.environ.get('PP3_CONFIG'):
        import torch
        from pp3.native_runtime import from_environment, write_json
        runtime = from_environment()
        settings = runtime.config['model']
        config = dict(GATE_ZO_DEFAULTS)
        config.pop('seed', None)
        config.update(selection_mode='gatezo', growth_protocol='validation_checkpoint',
            L_max=settings['epochs'], T_max=settings['pool_size'], nB=settings['growth_step'],
            gate_zo_min_keep=settings['zo_min_keep'], gate_zo_sparsity_lambda=settings['zo_sparsity_lambda'],
            gate_zo_sample_size=settings['zo_sample_size'] or None,
            gate_zo_calibration_ratio=settings['calibration_ratio'],
            gate_zo_validation_ratio=settings['validation_ratio'], gate_zo_ridge=settings['ridge'],
            gate_zo_growth_min_delta=settings['growth_min_delta'],
            gate_zo_failure_patience=settings['growth_failure_patience'],
            gate_zo_seed=runtime.config['seed'], compute_device=str(runtime.device),
            compute_batch_size=settings['batch_size'], zo_steps=0)
        write_json(runtime.output/'native_args.json', config)
        x, y, x_test, y_test = runtime.arrays()
        classes = runtime.config['data']['classes']
        targets = torch.nn.functional.one_hot(y, classes).float()
        test_targets = torch.nn.functional.one_hot(y_test, classes).float()
        runtime.begin('prepare', unknown='Native Python/NumPy initialization not fully counted')
        model = BDNN1GateZO(**config)
        model._pp3_runtime = runtime
        runtime.end()
        result = model.classification(x, targets, x_test, test_targets)
        runtime.finish(dict(W=result[4], U=result[5], Beta=result[6]))
        write_json(runtime.output/'native_summary.json', model.gate_zo_summary)
        return
    args = parse_args()''')
    entry.write_text(source, encoding='utf-8')
    path = package/'structural_gate.py'
    source = path.read_text(encoding='utf-8')
    assert '_pp3_runtime' not in source
    source = source.replace('    attempts = 0\n', "    runtime = getattr(model, '_pp3_runtime', None)\n    attempts = 0\n", 1)
    source = source.replace('        attempts += 1\n', "        attempts += 1\n        if runtime:\n            runtime.begin('train', attempts, unknown='NumPy candidate search, ridge, mask score and scalar arithmetic not fully counted')\n", 1)
    source = source.replace('        if flag == 1:\n            break', "        if flag == 1:\n            if runtime:\n                runtime.end()\n                runtime.epoch_row(attempts, unit='growth_round_failed')\n            break", 1)
    source = source.replace('        selector.history.append(record)\n', '''        selector.history.append(record)
        if runtime:
            runtime.end(train_metrics['accuracy'], train_metrics['mse'])
            runtime.begin('test', attempts, unknown='NumPy readout matrix multiply and metric arithmetic not counted')
            test_h, _ = model.getH(X_test)
            observed = classification_metrics(np.asarray(test_h), T_test, model.Beta)
            runtime.end(observed['accuracy'], observed['mse'])
            runtime.epoch_row(attempts, unit='growth_round')
            runtime.rows[-1]['validation_accuracy'] = checkpoint_metrics['accuracy']
            runtime._csv()
''', 1)
    source = source.replace('    selector.restore(model, best_state)\n', "    if runtime:\n        runtime.begin('final_refit_and_test', unknown='NumPy readout fit and evaluation arithmetic not fully counted')\n    selector.restore(model, best_state)\n", 1)
    source = source.replace('    model.COST = train_metrics["mse"]\n', "    if runtime:\n        runtime.end(test_metrics['accuracy'], test_metrics['mse'])\n    model.COST = train_metrics[\"mse\"]\n", 1)
    path.write_text(source, encoding='utf-8')
