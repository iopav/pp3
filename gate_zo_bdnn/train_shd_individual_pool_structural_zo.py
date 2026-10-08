"""SHD individual-pool BDNN-RSNN with one structural ZO step per growth.

BDNN first ranks and provisionally selects at most 16 independent neurons.  In
``gatezo`` mode, one coordinate zeroth-order step estimates every provisional
neuron's removal cost on an internal fit/calibration split.  The resulting
L0-penalized mask is applied before the outer speaker-validation growth gate.
No backpropagation is used and the SHD test set is never used for fitting,
mask selection, growth decisions, or stopping.
"""

from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path

import numpy as np
import torch

try:
    from . import train_shd_bdnn_rsnn_blocks as blocks
    from . import train_shd_bdnn_rsnn_pure as core
    from .shd_gatezo_support import (
        balanced_sample,
        dataset_labels,
        propose_individual_pool,
        select_residual_diverse_neurons,
        solve_sequence_ridge,
        apply_sequence_readout,
        temporal_window_features,
        profiled_ridge_metrics,
        stratified_fit_calibration_split,
    )
except ImportError:
    import train_shd_bdnn_rsnn_blocks as blocks
    import train_shd_bdnn_rsnn_pure as core
    from shd_gatezo_support import (
        balanced_sample,
        dataset_labels,
        propose_individual_pool,
        select_residual_diverse_neurons,
        solve_sequence_ridge,
        apply_sequence_readout,
        temporal_window_features,
        profiled_ridge_metrics,
        stratified_fit_calibration_split,
    )



def sequence_features(traces, args):
    pyramid = tuple(sorted(set(getattr(args, "temporal_pyramid", ()))))
    readout_taus = tuple(getattr(args, "readout_filter_tau_ms", ()))
    if pyramid or readout_taus:
        if args.temporal_pooling != "window_mean":
            raise ValueError("temporal pyramid requires window_mean pooling")
        selected = traces[:, args.readout_warmup_bins:, :]
        components = []
        pyramid_tensors = []
        for windows in pyramid or (args.temporal_windows,):
            if windows < 1 or windows > selected.shape[1]:
                raise ValueError("temporal pyramid windows must fit readout bins")
            bounds = np.linspace(0, selected.shape[1], windows + 1, dtype=np.int64)
            pooled = torch.stack([
                selected[:, int(bounds[i]):int(bounds[i + 1])].mean(1)
                for i in range(windows)
            ], dim=1)
            pyramid_tensors.append(pooled)
            components.append(pooled)
            if getattr(args, "temporal_differences", False) and windows > 1:
                delta = pooled[:, 1:] - pooled[:, :-1]
                components.extend([
                    delta.mean(1, keepdim=True),
                    delta.abs().mean(1, keepdim=True),
                    delta.square().mean(1, keepdim=True).sqrt(),
                ])
        if getattr(args, "global_temporal_stats", False):
            components.extend([
                selected.mean(1, keepdim=True),
                selected.std(1, unbiased=False, keepdim=True),
                selected.amax(1, keepdim=True),
            ])
        if readout_taus:
            bin_dt = float(args.max_time) / float(args.readout_bins)
            filter_windows = int(getattr(args, "readout_filter_windows", 8))
            bounds = np.linspace(
                0, selected.shape[1], filter_windows + 1, dtype=np.int64
            )
            for tau_ms in readout_taus:
                decay = float(np.exp(-bin_dt / (float(tau_ms) / 1000.0)))
                filtered = torch.zeros_like(selected)
                state = torch.zeros_like(selected[:, 0])
                for step in range(selected.shape[1]):
                    state = decay * state + (1.0 - decay) * selected[:, step]
                    filtered[:, step] = state
                components.append(torch.stack([
                    filtered[:, int(bounds[i]):int(bounds[i + 1])].mean(1)
                    for i in range(filter_windows)
                ], dim=1))
        component_tensor = torch.cat(components, dim=1)
        projection_count = int(getattr(args, "temporal_component_projection", 0))
        if projection_count and component_tensor.shape[1] > projection_count:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(args.random_readout_seed) + 17)
            projection = torch.randn(
                component_tensor.shape[1], projection_count,
                generator=generator, dtype=component_tensor.dtype,
            ) / np.sqrt(float(component_tensor.shape[1]))
            component_tensor = torch.einsum(
                "nch,cp->nph", component_tensor, projection
            )
        random_count = int(getattr(args, "random_readout_features", 0))
        if random_count:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(args.random_readout_seed))
            weights = torch.randn(
                component_tensor.shape[1], random_count, generator=generator,
                dtype=component_tensor.dtype,
            ) / np.sqrt(float(component_tensor.shape[1]))
            bias = torch.randn(
                random_count, generator=generator, dtype=component_tensor.dtype,
            ) * 0.25
            nonlinear = torch.tanh(
                torch.einsum("nch,cr->nrh", component_tensor, weights)
                + bias[None, :, None]
            )
            component_tensor = torch.cat([component_tensor, nonlinear], dim=1)
        return component_tensor.flatten(1)
    return temporal_window_features(
        traces,
        args.readout_warmup_bins,
        args.temporal_windows,
        pooling=args.temporal_pooling,
        attention_temperature=args.attention_temperature,
        attention_activity_mix=args.attention_activity_mix,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-mode", choices=("fixed16", "adaptive_nozo", "gatezo"),
        default="gatezo",
    )
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--readout-bins", type=int, default=100)
    parser.add_argument("--readout-warmup-bins", type=int, default=5)
    parser.add_argument("--temporal-windows", type=int, default=12)
    parser.add_argument(
        "--temporal-pooling",
        choices=("window_mean", "cumulative_sum", "parameter_free_attention_sum"),
        default="window_mean",
    )
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--attention-activity-mix", type=float, default=0.5)
    parser.add_argument("--temporal-pyramid", type=int, nargs="*", default=[])
    parser.add_argument("--temporal-differences", action="store_true")
    parser.add_argument("--global-temporal-stats", action="store_true")
    parser.add_argument("--readout-filter-tau-ms", type=float, nargs="*", default=[])
    parser.add_argument("--readout-filter-windows", type=int, default=8)
    parser.add_argument("--random-readout-features", type=int, default=0)
    parser.add_argument("--random-readout-seed", type=int, default=314159)
    parser.add_argument("--temporal-component-projection", type=int, default=0)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--layer2-candidate-pool-size", type=int, default=0)
    parser.add_argument("--layer1-target-hidden", type=int, default=0)
    parser.add_argument("--growth-step", type=int, default=16)
    parser.add_argument("--growth-choices", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument(
        "--max-hidden", type=int, default=512,
        help="maximum committed hidden neurons; 0 disables the hidden-size cap",
    )
    parser.add_argument("--max-growth-rounds", type=int, default=120)
    parser.add_argument("--growth-failure-patience", type=int, default=12)
    parser.add_argument("--growth-min-delta", type=float, default=0.002)
    parser.add_argument("--ridge", type=float, default=0.5)
    parser.add_argument("--input-scales", type=float, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--self-radii", type=float, nargs="+", default=[0.7, 0.9, 1.1, 1.3])
    parser.add_argument("--old-to-new-scale", type=float, default=0.5)
    parser.add_argument("--bias-scale", type=float, default=0.2)
    parser.add_argument("--candidate-min-rate", type=float, default=0.0005)
    parser.add_argument("--candidate-max-rate", type=float, default=0.30)
    parser.add_argument("--diversity-strength", type=float, default=0.5)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--tau-syn", type=float, default=0.010)
    parser.add_argument("--tau-mem", type=float, default=0.020)
    parser.add_argument("--heterogeneous-tau-fraction", type=float, default=0.0)
    parser.add_argument("--tau-syn-min", type=float, default=0.010)
    parser.add_argument("--tau-syn-max", type=float, default=0.010)
    parser.add_argument("--tau-mem-min", type=float, default=0.020)
    parser.add_argument("--tau-mem-max", type=float, default=0.020)
    parser.add_argument("--tau-adapt-min", type=float, default=0.060)
    parser.add_argument("--tau-adapt-max", type=float, default=0.200)
    parser.add_argument("--adapt-a-min", type=float, default=-0.5)
    parser.add_argument("--adapt-a-max", type=float, default=0.5)
    parser.add_argument("--adapt-b-min", type=float, default=0.05)
    parser.add_argument("--adapt-b-max", type=float, default=0.50)
    parser.add_argument(
        "--input-delay-ms", type=float, nargs="+", default=[0.0],
        help=(
            "candidate-level input delays in milliseconds; used only by "
            "dynamics backends that support frozen delays"
        ),
    )
    parser.add_argument("--input-channel-group-size", type=int, default=1)
    parser.add_argument("--train-channel-jitter-std", type=float, default=0.0)
    parser.add_argument("--train-jitter-copies", type=int, default=0)
    parser.add_argument("--train-time-scale-min", type=float, default=1.0)
    parser.add_argument("--train-time-scale-max", type=float, default=1.0)
    parser.add_argument("--input-branches", type=int, default=1)
    parser.add_argument("--deterministic-delay-bank", action="store_true")
    parser.add_argument("--layer2-raw-input-scale", type=float, default=0.25)
    parser.add_argument("--layer2-connectivity", type=float, default=0.20)
    parser.add_argument("--layer2-recurrent-delay-ms", type=float, nargs="*", default=[])
    parser.add_argument(
        "--frequency-bandwidths", type=int, nargs="+", default=[700]
    )
    parser.add_argument("--branch-tau-syn-ms", type=float, nargs="*", default=[])
    parser.add_argument(
        "--proposal-mode", choices=("prior", "residual", "shuffled", "activity"),
        default="prior",
        help="prior disables residual-adaptive stochastic proposals",
    )
    parser.add_argument("--guided-proposal-fraction", type=float, default=0.5)
    parser.add_argument("--proposal-time-windows", type=int, default=16)
    parser.add_argument("--proposal-ema-alpha", type=float, default=0.25)
    parser.add_argument("--proposal-uniform-floor", type=float, default=0.20)
    parser.add_argument("--proposal-fit-ratio", type=float, default=0.40)
    parser.add_argument("--proposal-oof-folds", type=int, default=5)
    parser.add_argument("--three-factor-mix", type=float, default=0.0)
    parser.add_argument("--eligibility-tau", type=float, default=0.050)
    parser.add_argument("--membrane-surrogate-width", type=float, default=0.25)
    parser.add_argument(
        "--delay-mutation-control",
        choices=("none", "directed", "matched_shuffled"), default="none",
    )
    parser.add_argument("--delay-direction-oof-folds", type=int, default=5)
    parser.add_argument("--delay-surrogate-gamma", type=float, default=0.3)
    parser.add_argument("--delay-variant-sample-size", type=int, default=1024)
    parser.add_argument("--delay-mutation-min-mismatch-rate", type=float, default=0.70)
    parser.add_argument("--delay-mutation-min-informative-rounds", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--validation-speakers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zo-sample-size", type=int, default=2048)
    parser.add_argument("--zo-calibration-ratio", type=float, default=0.20)
    parser.add_argument("--zo-sparsity-lambda", type=float, default=0.001)
    parser.add_argument("--zo-min-keep", type=int, default=4)
    parser.add_argument("--compact-loss-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="stop after structure validation; use this during hyperparameter search",
    )
    parser.add_argument("--resume", action="store_true")
    if os.environ.get('PP3_CONFIG'):
        config = json.loads(Path(os.environ['PP3_CONFIG']).read_text(encoding='utf-8'))
        model, data = config['model'], config['data']
        values = dict(model.get('native_args', {}))
        values.update({key: value for key, value in model.items()
                       if key in {action.dest for action in parser._actions}})
        values.update(data_dir=data['root'], output_dir=config['output'], seed=config['seed'],
            validation_speakers=config.get('validation', {}).get('speakers', 2),
            device=config['device'], num_workers=config['workers'], batch_size=model['batch_size'],
            candidate_pool_size=model['pool_size'], max_growth_rounds=model['epochs'],
            max_hidden=model.get('max_hidden', 0), dt=data['dt'], max_time=data['max_time'],
            train_channel_jitter_std=data['jitter_std'], train_jitter_copies=data['jitter_copies'],
            train_time_scale_min=data['time_scale_min'], train_time_scale_max=data['time_scale_max'])
        argv = []
        for action in parser._actions:
            if action.dest not in values:
                continue
            value = values[action.dest]
            if isinstance(action, argparse._StoreTrueAction):
                if value:
                    argv.append(action.option_strings[0])
            else:
                argv.extend([action.option_strings[0], *map(str, value if isinstance(value, list) else [value])])
        return parser.parse_args(argv)
    return parser.parse_args()


def load_state(saved):
    return core.RandomRSNNState(
        config=core.DynamicsConfig(**saved["config"]),
        input_weight=saved["input_weight"].float(),
        recurrent_weight=saved["recurrent_weight"].float(),
        bias=saved["bias"].float(),
    )


def mask_metrics(
    provisional_traces,
    labels,
    old_hidden,
    keep,
    fit_positions,
    calibration_positions,
    args,
    device,
):
    traces = core.selected_traces(provisional_traces, old_hidden, keep)
    features = sequence_features(traces, args)
    fit_positions = torch.as_tensor(fit_positions, dtype=torch.long)
    calibration_positions = torch.as_tensor(calibration_positions, dtype=torch.long)
    fit_metrics, calibration_metrics = profiled_ridge_metrics(
        features[fit_positions],
        labels[fit_positions],
        features[calibration_positions],
        labels[calibration_positions],
        args.ridge,
        device,
    )
    return {
        "keep": list(map(int, keep)),
        "fit": fit_metrics,
        "calibration": calibration_metrics,
        "loss": float(calibration_metrics["loss"]),
    }


def choose_adaptive_prefix(evaluations, tolerance):
    """Smallest prefix within ``tolerance`` of the best calibration loss."""
    best_loss = min(item["loss"] for item in evaluations)
    eligible = [item for item in evaluations if item["loss"] <= best_loss + tolerance]
    return min(eligible, key=lambda item: (len(item["keep"]), item["loss"]))


def choose_coordinate_zo_mask(full, ablations, sparsity_lambda, min_keep):
    """One coordinate-ZO structural update for an L0-penalized objective.

    ``loss_without_i - loss_full`` is the zero-order importance of neuron i.
    We estimate all coordinates once, order neurons by that estimate, and take
    the best L0-penalized prefix.  The caller supplies exact combined-prefix
    losses because individual removal effects need not be additive.
    """
    importance = {
        int(item["removed"]): float(item["loss"] - full["loss"])
        for item in ablations
    }
    order = sorted(importance, key=lambda index: (-importance[index], index))
    min_keep = min(max(1, int(min_keep)), len(order))
    return importance, order, min_keep


def internal_positions(labels, fit_pool, cal_pool, sample_size, rng):
    fit_count = (int(sample_size) + 1) // 2
    cal_count = int(sample_size) // 2
    return (
        balanced_sample(fit_pool, labels, fit_count, rng),
        balanced_sample(cal_pool, labels, cal_count, rng),
    )


def choose_new_neurons(
    provisional_traces,
    labels,
    old_hidden,
    selected_count,
    fit_positions,
    cal_positions,
    args,
    device,
):
    all_keep = list(range(selected_count))
    if args.selection_mode == "fixed16":
        return all_keep, {"mode": "fixed16", "selected": selected_count}

    if args.selection_mode == "adaptive_nozo":
        choices = sorted(
            set(min(selected_count, value) for value in args.growth_choices if value > 0)
        )
        if selected_count not in choices:
            choices.append(selected_count)
        evaluations = [
            mask_metrics(
                provisional_traces, labels, old_hidden, list(range(count)),
                fit_positions, cal_positions, args, device,
            )
            for count in choices
        ]
        chosen = choose_adaptive_prefix(evaluations, args.compact_loss_tolerance)
        return chosen["keep"], {
            "mode": "adaptive_nozo",
            "evaluations": evaluations,
            "chosen_loss": chosen["loss"],
        }

    full = mask_metrics(
        provisional_traces, labels, old_hidden, all_keep,
        fit_positions, cal_positions, args, device,
    )
    ablations = []
    for removed in all_keep:
        item = mask_metrics(
            provisional_traces,
            labels,
            old_hidden,
            [index for index in all_keep if index != removed],
            fit_positions,
            cal_positions,
            args,
            device,
        )
        item["removed"] = removed
        ablations.append(item)
    importance, order, min_keep = choose_coordinate_zo_mask(
        full, ablations, args.zo_sparsity_lambda, args.zo_min_keep
    )
    candidates = [full]
    for count in range(min_keep, selected_count):
        candidates.append(
            mask_metrics(
                provisional_traces, labels, old_hidden, order[:count],
                fit_positions, cal_positions, args, device,
            )
        )
    chosen = min(
        candidates,
        key=lambda item: (
            item["loss"] + args.zo_sparsity_lambda * len(item["keep"]),
            len(item["keep"]),
        ),
    )
    return chosen["keep"], {
        "mode": "gatezo",
        "dimension": selected_count,
        "directions": selected_count,
        "finite_difference": "one-sided hard coordinate ablation",
        "fit_samples": len(fit_positions),
        "calibration_samples": len(cal_positions),
        "sparsity_lambda": args.zo_sparsity_lambda,
        "full_loss": full["loss"],
        "importance": [importance[index] for index in range(selected_count)],
        "importance_order": order,
        "chosen_loss": chosen["loss"],
        "chosen_penalized_objective": (
            chosen["loss"] + args.zo_sparsity_lambda * len(chosen["keep"])
        ),
    }


def main():
    args = parse_args()
    runtime = None
    if os.environ.get('PP3_CONFIG'):
        from pp3.native_runtime import from_environment, MeasuredLoader, write_json
        runtime = from_environment()
        if runtime.config.get('smoke'):
            args.pp3_smoke_samples = runtime.config.get('smoke_samples', 4)
        write_json(runtime.output/'native_args.json', vars(args))
    if not 0.0 < args.zo_calibration_ratio < 1.0:
        raise ValueError("zo-calibration-ratio must be in (0, 1)")
    if args.growth_step <= 0 or args.zo_min_keep <= 0:
        raise ValueError("growth-step and zo-min-keep must be positive")
    if args.max_hidden < 0:
        raise ValueError("max-hidden must be nonnegative (0 means unlimited)")
    if args.input_channel_group_size <= 0:
        raise ValueError("input-channel-group-size must be positive")
    if not args.input_delay_ms or any(value < 0.0 for value in args.input_delay_ms):
        raise ValueError("input-delay-ms must contain nonnegative values")
    if args.train_channel_jitter_std < 0.0 or args.train_jitter_copies < 0:
        raise ValueError("jitter settings cannot be negative")
    if (args.train_channel_jitter_std > 0.0) != (args.train_jitter_copies > 0):
        raise ValueError("positive jitter std and copies must be enabled together")
    if (
        args.train_time_scale_min <= 0.0
        or args.train_time_scale_max < args.train_time_scale_min
    ):
        raise ValueError("invalid training time-scale range")
    if args.input_branches < 1 or any(value < 1 for value in args.frequency_bandwidths):
        raise ValueError("invalid branch or frequency-bandwidth configuration")
    if any(value < 1 for value in args.temporal_pyramid):
        raise ValueError("temporal-pyramid entries must be positive")
    if any(value <= 0 for value in args.readout_filter_tau_ms):
        raise ValueError("readout filter taus must be positive")
    if (
        args.readout_filter_windows < 1
        or args.random_readout_features < 0
        or args.temporal_component_projection < 0
    ):
        raise ValueError("invalid readout feature settings")
    if args.layer1_target_hidden < 0 or args.layer2_candidate_pool_size < 0:
        raise ValueError("invalid hierarchy settings")
    if args.layer2_raw_input_scale < 0 or not 0.0 < args.layer2_connectivity <= 1.0:
        raise ValueError("invalid layer-2 connectivity settings")
    if any(value < 0 for value in args.layer2_recurrent_delay_ms):
        raise ValueError("layer-2 recurrent delays cannot be negative")
    if not 0.0 < args.proposal_fit_ratio < 1.0 or args.proposal_oof_folds < 2:
        raise ValueError("invalid proposal split or OOF fold count")
    if not 0.0 <= args.three_factor_mix <= 1.0:
        raise ValueError("three-factor-mix must be in [0,1]")
    if args.eligibility_tau <= 0.0 or args.membrane_surrogate_width <= 0.0:
        raise ValueError("invalid three-factor eligibility settings")
    if args.delay_direction_oof_folds < 2 or args.delay_variant_sample_size < 2:
        raise ValueError("invalid delay-mutation sampling settings")
    if not 0.0 <= args.delay_mutation_min_mismatch_rate <= 1.0:
        raise ValueError("invalid delay-mutation mismatch threshold")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        _, _, train_loader, val_loader, test_loader, _, _, held_out_speakers
    ) = blocks.build_speaker_loaders(args)
    if runtime:
        train_loader, val_loader, test_loader = [MeasuredLoader(loader, runtime)
            for loader in (train_loader, val_loader, test_loader)]
        runtime.begin('prepare', unknown='NumPy proposal/partition and host scalar arithmetic not fully counted')
    labels_by_position = dataset_labels(train_loader.dataset)
    proposal_pool = ()
    structural_partitions = None
    if hasattr(core, "build_structural_partitions"):
        structural_partitions = core.build_structural_partitions(
            train_loader.dataset, labels_by_position, args.seed + 4049
        )
        fit_pool = tuple(sorted(
            structural_partitions["direction_fit"]
            + structural_partitions["variant_cal"]
        ))
        cal_pool = tuple(sorted(structural_partitions["gate_cal"]))
    elif hasattr(core, "proposal_partitions"):
        proposal_pool, fit_pool, cal_pool = core.proposal_partitions(
            labels_by_position, args.proposal_fit_ratio,
            args.zo_calibration_ratio, args.seed + 4049,
        )
    else:
        fit_pool, cal_pool = stratified_fit_calibration_split(
            labels_by_position, args.zo_calibration_ratio, args.seed + 4049
        )
    latest = args.output_dir / "latest_growth.pt"
    history, failures, start_round = [], 0, 1
    if args.resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu")
        if saved.get("structural_partitions") != structural_partitions:
            raise RuntimeError("resume structural partitions do not match")
        state = load_state(saved["state"])
        history = saved.get("history", [])
        failures = int(saved.get("failures", 0))
        start_round = int(saved.get("next_round", len(history) + 1))
        print(f"[resume] hidden={state.hidden_size} next_round={start_round} failures={failures}")
    else:
        dynamics_kwargs = {
            "input_size": (700 + args.input_channel_group_size - 1)
            // args.input_channel_group_size,
            "dt": args.dt,
            "tau_syn": args.tau_syn,
            "tau_mem": args.tau_mem,
            "threshold": args.threshold,
        }
        if "input_delay_ms" in getattr(core.DynamicsConfig, "__dataclass_fields__", {}):
            dynamics_kwargs["input_delay_ms"] = tuple(args.input_delay_ms)
        optional_dynamics = {
            "heterogeneous_tau_fraction": args.heterogeneous_tau_fraction,
            "tau_syn_min": args.tau_syn_min,
            "tau_syn_max": args.tau_syn_max,
            "tau_mem_min": args.tau_mem_min,
            "tau_mem_max": args.tau_mem_max,
            "tau_adapt_min": args.tau_adapt_min,
            "tau_adapt_max": args.tau_adapt_max,
            "adapt_a_min": args.adapt_a_min,
            "adapt_a_max": args.adapt_a_max,
            "adapt_b_min": args.adapt_b_min,
            "adapt_b_max": args.adapt_b_max,
            "input_branches": args.input_branches,
            "frequency_bandwidths": tuple(args.frequency_bandwidths),
            "branch_tau_syn_ms": tuple(args.branch_tau_syn_ms),
            "deterministic_delay_bank": args.deterministic_delay_bank,
            "layer1_target_hidden": args.layer1_target_hidden,
            "layer2_raw_input_scale": args.layer2_raw_input_scale,
            "layer2_connectivity": args.layer2_connectivity,
            "layer2_recurrent_delay_ms": tuple(args.layer2_recurrent_delay_ms),
            "proposal_mode": args.proposal_mode,
            "guided_proposal_fraction": args.guided_proposal_fraction,
            "proposal_time_windows": args.proposal_time_windows,
            "proposal_ema_alpha": args.proposal_ema_alpha,
            "proposal_uniform_floor": args.proposal_uniform_floor,
        }
        fields = getattr(core.DynamicsConfig, "__dataclass_fields__", {})
        dynamics_kwargs.update(
            {key: value for key, value in optional_dynamics.items() if key in fields}
        )
        state = core.RandomRSNNState.empty(core.DynamicsConfig(**dynamics_kwargs))

    train_traces, train_labels, _ = core.extract_readout_traces(
        train_loader, state, device, args.readout_bins
    )
    val_traces, val_labels, _ = core.extract_readout_traces(
        val_loader, state, device, args.readout_bins
    )
    train_features = sequence_features(train_traces, args)
    val_features = sequence_features(val_traces, args)
    beta, current_train, current_val, feature_mean, feature_scale = solve_sequence_ridge(
        train_features, train_labels, val_features, val_labels, args.ridge, device
    )
    print(
        f"structural-ZO mode={args.selection_mode} seed={args.seed} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"preselect={args.growth_step} zo_samples={args.zo_sample_size} "
        f"lambda={args.zo_sparsity_lambda} hidden={state.hidden_size} "
        f"max_hidden={'unlimited' if args.max_hidden <= 0 else args.max_hidden} "
        f"input_channels={state.config.input_size} "
        f"jitter_std={args.train_channel_jitter_std} "
        f"jitter_copies={args.train_jitter_copies} "
        f"val_acc={current_val['accuracy']:.4f}", flush=True,
    )

    if runtime:
        runtime.end()
    for round_index in range(start_round, args.max_growth_rounds + 1):
        capacity = (
            args.max_hidden - state.hidden_size
            if args.max_hidden > 0
            else args.growth_step
        )
        if args.max_hidden > 0 and capacity <= 0:
            break
        if runtime:
            runtime.begin('train', round_index, unknown='NumPy proposals and host scalar arithmetic not counted; solve operators audited individually')
        old_hidden = state.hidden_size
        candidate_rng = np.random.RandomState(args.seed + round_index * 1009)
        round_pool_size = (
            core.candidate_pool_size(state, args)
            if hasattr(core, "candidate_pool_size")
            else args.candidate_pool_size
        )
        pool_state, metadata = propose_individual_pool(
            state, round_pool_size, args.input_scales, args.self_radii,
            args.old_to_new_scale, args.bias_scale, candidate_rng,
        )
        three_factor_eligibility = None
        if args.three_factor_mix > 0.0:
            if not hasattr(core, "extract_readout_traces_with_eligibility"):
                raise RuntimeError("selected dynamics backend lacks three-factor support")
            pool_train, check, rates, three_factor_eligibility = (
                core.extract_readout_traces_with_eligibility(
                    train_loader, pool_state, device, args.readout_bins,
                    args.readout_warmup_bins,
                    old_hidden, args.temporal_windows, args.eligibility_tau,
                    args.membrane_surrogate_width,
                )
            )
        else:
            pool_train, check, rates = core.extract_readout_traces(
                train_loader, pool_state, device, args.readout_bins
            )
        if not torch.equal(check, train_labels):
            raise RuntimeError("training sample order changed")
        args.current_feature_mean = feature_mean
        args.current_feature_scale = feature_scale
        selected, screen = select_residual_diverse_neurons(
            train_features, beta, pool_train, train_labels, old_hidden,
            rates[old_hidden:], args, three_factor_eligibility,
        )
        selected = selected[: min(capacity, args.growth_step)]
        keep, selection_record = [], {"mode": args.selection_mode}
        prepared = None
        if selected:
            if hasattr(core, "prepare_structural_candidates"):
                prepared = core.prepare_structural_candidates(
                    train_loader, pool_state, pool_train, train_labels,
                    selected, old_hidden, structural_partitions, args, device,
                    round_index,
                )
                provisional_state = prepared["state"]
                provisional_traces = prepared["traces"]
            else:
                provisional_state = pool_state
                provisional_traces = core.selected_traces(
                    pool_train, old_hidden, selected
                )
            sample_rng = np.random.RandomState(args.seed + round_index * 1009 + 7919)
            fit_positions, cal_positions = internal_positions(
                labels_by_position, fit_pool, cal_pool, args.zo_sample_size, sample_rng
            )
            keep, selection_record = choose_new_neurons(
                provisional_traces, train_labels, old_hidden, len(selected),
                fit_positions, cal_positions, args, device,
            )
        committed_candidates = [selected[index] for index in keep]
        if prepared is not None:
            proposal = provisional_state.select_from_pool(old_hidden, keep)
        else:
            proposal = pool_state.select_from_pool(old_hidden, committed_candidates)
        if prepared is not None:
            proposal_train_traces = core.selected_traces(
                provisional_traces, old_hidden, keep
            )
        else:
            proposal_train_traces = core.selected_traces(
                pool_train, old_hidden, committed_candidates
            )
        proposal_val_traces, vcheck, _ = core.extract_readout_traces(
            val_loader, proposal, device, args.readout_bins
        )
        if not torch.equal(vcheck, val_labels):
            raise RuntimeError("validation sample order changed")
        proposal_train_features = sequence_features(proposal_train_traces, args)
        proposal_val_features = sequence_features(proposal_val_traces, args)
        proposal_beta, proposal_train, proposal_val, proposal_mean, proposal_scale = (
            solve_sequence_ridge(
                proposal_train_features, train_labels, proposal_val_features,
                val_labels, args.ridge, device,
            )
        )
        accepted = bool(committed_candidates) and proposal_val["accuracy"] > (
            current_val["accuracy"] + args.growth_min_delta
        )
        history.append(
            {
                "round": round_index,
                "old_hidden": old_hidden,
                "preselected": selected,
                "preselected_metadata": [metadata[index] for index in selected],
                "screen": screen,
                "structural_selection": selection_record,
                "kept_relative": keep,
                "kept_candidates": committed_candidates,
                "candidate_preparation": (
                    prepared["record"] if prepared is not None else None
                ),
                "proposal_validation": proposal_val,
                "accepted": bool(accepted),
            }
        )
        if accepted:
            state = proposal
            train_traces, val_traces = proposal_train_traces, proposal_val_traces
            train_features, val_features = proposal_train_features, proposal_val_features
            beta, current_train, current_val = proposal_beta, proposal_train, proposal_val
            feature_mean, feature_scale = proposal_mean, proposal_scale
            if hasattr(core, "update_proposal_distribution"):
                state, proposal_update = core.update_proposal_distribution(
                    state, train_loader, train_features, train_labels,
                    proposal_pool, args, device, round_index,
                )
                history[-1]["proposal_distribution_update"] = proposal_update
            failures = 0
            print(
                f"[commit] round={round_index} preselected={len(selected)} "
                f"kept={len(keep)} hidden={state.hidden_size} "
                f"val_acc={current_val['accuracy']:.4f}", flush=True,
            )
        else:
            failures += 1
            print(
                f"[rollback] round={round_index} preselected={len(selected)} "
                f"kept={len(keep)} proposal_val={proposal_val['accuracy']:.4f} "
                f"base_val={current_val['accuracy']:.4f} "
                f"failures={failures}/{args.growth_failure_patience}", flush=True,
            )
        if runtime:
            runtime.end(current_train['accuracy'])
            runtime.begin('test', round_index, unknown='Native host scalar metrics not fully audited')
            observed_traces, observed_labels, _ = core.extract_readout_traces(
                test_loader, state, device, args.readout_bins)
            observed_features = sequence_features(observed_traces, args)
            observed_logits = apply_sequence_readout(observed_features, beta, feature_mean, feature_scale)
            observed_metrics = blocks.core.classification_metrics(observed_logits, observed_labels)
            runtime.end(observed_metrics['accuracy'])
            # Monitoring only: no test quantity feeds the growth decision or final refit.
            runtime.epoch_row(round_index, unit='growth_round')
            runtime.rows[-1]['validation_accuracy'] = current_val['accuracy']
            runtime._csv()
        torch.save(
            {
                "state": state.checkpoint(), "history": history,
                "failures": failures, "next_round": round_index + 1,
                "selection_mode": args.selection_mode,
                "structural_partitions": structural_partitions,
            }, latest,
        )
        if failures >= args.growth_failure_patience:
            break

    if runtime:
        runtime.begin('final_refit_and_test', unknown='Host scalar arithmetic and native solve coverage incomplete')
    if args.skip_test:
        full_metrics = None
        test_metrics = None
    else:
        test_traces, test_labels, _ = core.extract_readout_traces(
            test_loader, state, device, args.readout_bins
        )
        test_features = sequence_features(test_traces, args)
        full_features = torch.cat([train_features, val_features], 0)
        full_labels = torch.cat([train_labels, val_labels], 0)
        beta, full_metrics, test_metrics, feature_mean, feature_scale = solve_sequence_ridge(
            full_features, full_labels, test_features, test_labels, args.ridge, device
        )
    checkpoint = args.output_dir / "final_structural_zo.pt"
    if runtime:
        runtime.end(None if test_metrics is None else test_metrics['accuracy'])
    torch.save(
        {
            "state": state.checkpoint(), "beta": beta,
            "feature_mean": feature_mean, "feature_scale": feature_scale,
            "args": vars(args), "history": history,
            "structural_partitions": structural_partitions,
        }, checkpoint,
    )
    accepted = [item for item in history if item["accepted"]]
    if runtime:
        runtime.finish(dict(state=state.checkpoint(), beta=beta,
                            feature_mean=feature_mean, feature_scale=feature_scale))
    summary = {
        "selection_mode": args.selection_mode,
        "hidden_size": state.hidden_size,
        "structure_training": current_train,
        "structure_validation": current_val,
        "full_train_refit": full_metrics,
        "test": test_metrics,
        "accepted_rounds": len(accepted),
        "attempted_rounds": len(history),
        "mean_kept_per_accepted_round": (
            float(np.mean([len(item["kept_relative"]) for item in accepted]))
            if accepted else 0.0
        ),
        "total_preselected_on_accepted_rounds": sum(
            len(item["preselected"]) for item in accepted
        ),
        "total_pruned_on_accepted_rounds": sum(
            len(item["preselected"]) - len(item["kept_relative"]) for item in accepted
        ),
        "zo_steps_per_growth": 1 if args.selection_mode == "gatezo" else 0,
        "zo_coordinate_queries_per_growth": (
            args.growth_step if args.selection_mode == "gatezo" else 0
        ),
        "held_out_speakers": held_out_speakers,
        "structure_train_samples": len(train_loader.dataset),
        "structure_validation_samples": len(val_loader.dataset),
        "input_channels": state.config.input_size,
        "input_channel_group_size": args.input_channel_group_size,
        "train_channel_jitter_std": args.train_channel_jitter_std,
        "train_jitter_copies": args.train_jitter_copies,
        "train_time_scale_min": args.train_time_scale_min,
        "train_time_scale_max": args.train_time_scale_max,
        "tau_syn": args.tau_syn,
        "tau_mem": args.tau_mem,
        "input_delay_ms": list(args.input_delay_ms),
        "temporal_pooling": args.temporal_pooling,
        "temporal_pyramid": list(args.temporal_pyramid),
        "temporal_differences": bool(args.temporal_differences),
        "global_temporal_stats": bool(args.global_temporal_stats),
        "readout_filter_tau_ms": list(args.readout_filter_tau_ms),
        "readout_filter_windows": args.readout_filter_windows,
        "random_readout_features": args.random_readout_features,
        "temporal_component_projection": args.temporal_component_projection,
        "deterministic_delay_bank": bool(args.deterministic_delay_bank),
        "layer1_target_hidden": args.layer1_target_hidden,
        "layer2_candidate_pool_size": args.layer2_candidate_pool_size,
        "attention_temperature": args.attention_temperature,
        "attention_activity_mix": args.attention_activity_mix,
        "checkpoint": str(checkpoint),
        "validation_only": bool(args.skip_test),
        "test_used_for_fitting_or_selection": False,
        "structure_validation_used_for_structural_zo": False,
        "three_factor_mix": args.three_factor_mix,
        "eligibility_tau": args.eligibility_tau,
        "membrane_surrogate_width": args.membrane_surrogate_width,
    }
    if hasattr(state, "proposal_updates"):
        summary["proposal_mode"] = args.proposal_mode
        summary["proposal_updates"] = int(state.proposal_updates)
        summary["proposal_fit_samples"] = len(proposal_pool)
        summary["proposal_uses_validation_or_test"] = False
    preparation_records = [
        item.get("candidate_preparation") for item in history
        if item.get("candidate_preparation") is not None
    ]
    if preparation_records:
        informative = [row for row in preparation_records if row["informative"]]
        summary["delay_mutation_control"] = args.delay_mutation_control
        summary["delay_mutation_informative_rounds"] = len(informative)
        summary["delay_mutation_control_valid"] = bool(
            len(informative) >= args.delay_mutation_min_informative_rounds
            and all(
                row["nonzero_mismatch_rate"]
                >= args.delay_mutation_min_mismatch_rate
                for row in informative
            )
        )
        summary["delay_mutation_shadow_win_rate"] = float(
            np.mean([
                row["shadow_wins"] / max(1, row["pairs"])
                for row in preparation_records
            ])
        )
    if hasattr(state, "input_delay_steps"):
        delay_counts = {}
        for delay_steps in state.input_delay_steps.tolist():
            delay_ms = float(delay_steps) * args.dt * 1000.0
            key = f"{delay_ms:g}"
            delay_counts[key] = delay_counts.get(key, 0) + 1
        summary["committed_input_delay_counts_ms"] = delay_counts
    if hasattr(state, "branch_delay_steps"):
        delay_counts = {}
        for delay_steps in state.branch_delay_steps.flatten().tolist():
            delay_ms = float(delay_steps) * args.dt * 1000.0
            key = f"{delay_ms:g}"
            delay_counts[key] = delay_counts.get(key, 0) + 1
        summary["committed_branch_delay_counts_ms"] = delay_counts
        summary["input_branches"] = int(state.config.input_branches)
        if getattr(state.config, "branch_tau_syn_ms", ()):
            summary["committed_branch_tau_syn_ms"] = {
                "choices": list(state.config.branch_tau_syn_ms),
                "min": float(state.branch_tau_syn_values.min()) * 1000.0,
                "mean": float(state.branch_tau_syn_values.mean()) * 1000.0,
                "max": float(state.branch_tau_syn_values.max()) * 1000.0,
            }
    if hasattr(state, "tau_syn_values") and state.hidden_size:
        summary["committed_tau_syn_ms"] = {
            "min": float(state.tau_syn_values.min()) * 1000.0,
            "mean": float(state.tau_syn_values.mean()) * 1000.0,
            "max": float(state.tau_syn_values.max()) * 1000.0,
        }
        summary["committed_tau_mem_ms"] = {
            "min": float(state.tau_mem_values.min()) * 1000.0,
            "mean": float(state.tau_mem_values.mean()) * 1000.0,
            "max": float(state.tau_mem_values.max()) * 1000.0,
        }
    if hasattr(state, "layer_ids"):
        summary["committed_layer_counts"] = {
            str(layer): int(torch.sum(state.layer_ids == layer))
            for layer in torch.unique(state.layer_ids).tolist()
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
