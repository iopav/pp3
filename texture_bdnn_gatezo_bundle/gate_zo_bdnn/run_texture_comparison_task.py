"""Run one matched two-stage texture experiment for one method and seed."""

import argparse
import json
from pathlib import Path

import numpy as np

from inputsdata import load_data
from seed_utils import set_global_seed

from .bdnn1_gatezo import BDNN1GateZO
from .bdnn2_gatezo import BDNN2GateZO
from .config_gatezo import GATE_ZO_DEFAULTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-mode",
        choices=("fixed_nozo", "adaptive_nozo", "gatezo"), required=True,
    )
    parser.add_argument(
        "--growth-protocol",
        choices=("legacy_test", "validation_checkpoint"), required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--growth-step", type=int, default=3)
    parser.add_argument("--max-hidden", type=int, default=200)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--zo-sparsity-lambda", type=float, default=1e-4)
    parser.add_argument("--zo-loss-tolerance", type=float, default=None)
    parser.add_argument("--adaptive-tolerance", type=float, default=1e-3)
    parser.add_argument("--zo-sample-size", type=int, default=0)
    parser.add_argument("--calibration-ratio", type=float, default=0.5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--growth-min-delta", type=float, default=1e-4,
        help="minimum outer validation-loss improvement required to commit",
    )
    parser.add_argument("--failure-patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-batch-size", type=int, default=128)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def model_config(args):
    config = dict(GATE_ZO_DEFAULTS)
    config.pop("seed", None)
    config.update({
        "selection_mode": args.selection_mode,
        "growth_protocol": args.growth_protocol,
        "T_max": args.candidate_pool_size,
        "nB": args.growth_step,
        "L_max": args.max_hidden,
        "gate_zo_min_keep": args.min_keep,
        "gate_zo_sparsity_lambda": args.zo_sparsity_lambda,
        "gate_zo_loss_tolerance": args.zo_loss_tolerance,
        "adaptive_nozo_tolerance": args.adaptive_tolerance,
        "gate_zo_sample_size": args.zo_sample_size or None,
        "gate_zo_calibration_ratio": args.calibration_ratio,
        "gate_zo_validation_ratio": args.validation_ratio,
        "gate_zo_ridge": args.ridge,
        "gate_zo_growth_min_delta": args.growth_min_delta,
        "gate_zo_failure_patience": args.failure_patience,
        "gate_zo_seed": args.seed,
        "gate_zo_verbose": not args.quiet,
        "compute_device": args.device,
        "compute_batch_size": args.compute_batch_size,
        "zo_steps": 0,
    })
    return config


def load_stage(data_root, stage):
    if stage == 1:
        directory, suffix, offset = data_root / "old6classes", "old", 0
    else:
        directory, suffix, offset = data_root / "new6classes", "new", 6
    paths = (
        directory / "X_train_6cla_{}.npy".format(suffix),
        directory / "Y_train_6cla_{}.npy".format(suffix),
        directory / "X_test_6cla_{}.npy".format(suffix),
        directory / "Y_test_6cla_{}.npy".format(suffix),
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing texture data: {}".format(", ".join(missing)))
    return load_data(*paths, offset, 6)


def serializable_summary(model, seed, args, stage):
    return {
        "seed": seed,
        "stage": stage,
        "selection_mode": args.selection_mode,
        "growth_protocol": args.growth_protocol,
        "candidate_pool_size": args.candidate_pool_size,
        "growth_step": args.growth_step,
        "min_keep": args.min_keep,
        "zo_sparsity_lambda": args.zo_sparsity_lambda,
        "zo_loss_tolerance": args.zo_loss_tolerance,
        "adaptive_tolerance": args.adaptive_tolerance,
        **model.gate_zo_summary,
        "history": model.gate_zo_history,
    }


def save_stage(directory, model, result, summary):
    directory.mkdir(parents=True, exist_ok=True)
    errors, train_rates, test_rates, times, W, U, beta = result
    np.save(directory / "W.npy", W)
    np.save(directory / "U.npy", U)
    np.save(directory / "Beta.npy", beta)
    np.save(directory / "errors.npy", errors)
    np.save(directory / "train_rates.npy", train_rates)
    np.save(directory / "test_rates.npy", test_rates)
    np.save(directory / "time_hours.npy", times)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
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
            gate_zo_validation_ratio=runtime.config.get('validation', {}).get('ratio', .2), gate_zo_ridge=settings['ridge'],
            gate_zo_growth_min_delta=settings['growth_min_delta'],
            gate_zo_failure_patience=settings['growth_failure_patience'],
            gate_zo_seed=runtime.config['seed'], compute_device=str(runtime.device),
            compute_batch_size=settings['batch_size'], zo_steps=0)
        write_json(runtime.output/'native_args.json', config)
        x, y, x_test, y_test = runtime.arrays()
        classes = runtime.config['data']['classes']
        targets = torch.nn.functional.one_hot(y, classes).float()
        test_targets = torch.nn.functional.one_hot(y_test, classes).float()
        runtime.begin('prepare')
        model = BDNN1GateZO(**config)
        model._pp3_runtime = runtime
        runtime.end()
        result = model.classification(x, targets, x_test, test_targets)
        runtime.finish(dict(W=result[4], U=result[5], Beta=result[6]))
        write_json(runtime.output/'native_summary.json', model.gate_zo_summary)
        return
    args = parse_args()
    set_global_seed(args.seed)
    destination = (
        args.output_root / args.growth_protocol /
        args.selection_mode / "seed_{}".format(args.seed)
    )
    config = model_config(args)

    X1, T1, X1_test, T1_test = load_stage(args.data_root, 1)
    stage1 = BDNN1GateZO(**config)
    result1 = stage1.classification(X1, T1, X1_test, T1_test)
    summary1 = serializable_summary(stage1, args.seed, args, 1)
    save_stage(destination / "exp1", stage1, result1, summary1)
    print("STAGE1", json.dumps(stage1.gate_zo_summary, sort_keys=True))

    # Reset the global candidate generator so paired methods share the same
    # stage-2 seed as well as the same stage-1 seed.
    set_global_seed(args.seed + 1_000_003)
    X2, T2, X2_test, T2_test = load_stage(args.data_root, 2)
    stage2 = BDNN2GateZO(**config)
    result2 = stage2.classification(
        X2, T2, X2_test, T2_test,
        destination / "exp1/W.npy", destination / "exp1/U.npy",
    )
    summary2 = serializable_summary(stage2, args.seed, args, 2)
    save_stage(destination / "exp2", stage2, result2, summary2)
    print("STAGE2", json.dumps(stage2.gate_zo_summary, sort_keys=True))

    combined = {
        "seed": args.seed,
        "selection_mode": args.selection_mode,
        "growth_protocol": args.growth_protocol,
        "exp1": {key: summary1[key] for key in (
            "hidden_size", "structure_validation", "training", "test",
            "accepted_rounds", "attempted_rounds", "test_used_for_selection",
        )},
        "exp2": {key: summary2[key] for key in (
            "hidden_size", "structure_validation", "training", "test",
            "accepted_rounds", "attempted_rounds", "test_used_for_selection",
        )},
    }
    (destination / "summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(combined, sort_keys=True))


if __name__ == "__main__":
    main()
