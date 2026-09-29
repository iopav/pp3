"""Map one Slurm task to one adLIF-GateZO configuration and seed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .shd_structural_zo_search_space import load_configs
except ImportError:
    from shd_structural_zo_search_space import load_configs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--expected-configs", type=int, default=2)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_configs(args.config_file)
    if len(configs) != args.expected_configs:
        raise RuntimeError(
            f"expected {args.expected_configs} configs but found {len(configs)}"
        )
    config_index = args.task_id % len(configs)
    seed_index = args.task_id // len(configs)
    if seed_index >= len(args.seeds):
        raise IndexError("task-id exceeds config x seed grid")
    config = configs[config_index]
    seed = args.seeds[seed_index]
    trainer_script = config.get("trainer_script", "train_shd_adlif_gatezo.py")
    allowed_trainers = {
        "train_shd_adlif_gatezo.py",
        "train_shd_radlif_delay_gatezo.py",
        "train_shd_radlif_advanced_gatezo.py",
        "train_shd_radlif_residual_proposal.py",
        "train_shd_radlif_three_factor_delay_mutation.py",
        "train_shd_hierarchical_delayline_gatezo.py",
    }
    if trainer_script not in allowed_trainers:
        raise ValueError(f"unsupported trainer_script: {trainer_script}")
    output_dir = args.output_root / config["config_id"] / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_config.json").write_text(
        json.dumps({**config, "seed": seed}, indent=2)
    )

    command = [
        sys.executable, "-u",
        str(Path(__file__).with_name(trainer_script)),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(output_dir),
        "--selection-mode", config["selection_mode"],
        "--dt", "0.0005",
        "--max-time", "1.0",
        "--readout-bins", "100",
        "--readout-warmup-bins", "5",
        "--temporal-windows", str(config["temporal_windows"]),
        "--temporal-pooling", config.get("temporal_pooling", "window_mean"),
        "--attention-temperature", str(config.get("attention_temperature", 1.0)),
        "--attention-activity-mix", str(config.get("attention_activity_mix", 0.5)),
        "--temporal-pyramid",
        *[str(value) for value in config.get("temporal_pyramid", [])],
        *(["--temporal-differences"] if config.get("temporal_differences", False) else []),
        *(["--global-temporal-stats"] if config.get("global_temporal_stats", False) else []),
        "--readout-filter-tau-ms",
        *[str(value) for value in config.get("readout_filter_tau_ms", [])],
        "--readout-filter-windows", str(config.get("readout_filter_windows", 8)),
        "--random-readout-features", str(config.get("random_readout_features", 0)),
        "--random-readout-seed", str(config.get("random_readout_seed", 314159)),
        "--temporal-component-projection", str(config.get("temporal_component_projection", 0)),
        "--candidate-pool-size", str(config["candidate_pool_size"]),
        "--layer2-candidate-pool-size", str(config.get("layer2_candidate_pool_size", 0)),
        "--layer1-target-hidden", str(config.get("layer1_target_hidden", 0)),
        "--growth-step", str(config["growth_step"]),
        "--growth-choices", "4", "8", "12", "16",
        "--max-hidden", str(config["max_hidden"]),
        "--max-growth-rounds", "120",
        "--growth-failure-patience", str(config["growth_failure_patience"]),
        "--growth-min-delta", str(config["growth_min_delta"]),
        "--ridge", str(config["ridge"]),
        "--input-scales", "2.0", "3.0", "4.0", "6.0",
        "--self-radii", "0.7", "0.9", "1.1", "1.3",
        "--old-to-new-scale", "0.5",
        "--bias-scale", "0.20",
        "--candidate-min-rate", "0.0005",
        "--candidate-max-rate", "0.30",
        "--diversity-strength", str(config["diversity_strength"]),
        "--min-candidate-score", "0.0",
        "--tau-syn", str(config["tau_syn"]),
        "--tau-mem", str(config["tau_mem"]),
        "--heterogeneous-tau-fraction", str(config.get("heterogeneous_tau_fraction", 0.0)),
        "--tau-syn-min", str(config.get("tau_syn_min", config["tau_syn"])),
        "--tau-syn-max", str(config.get("tau_syn_max", config["tau_syn"])),
        "--tau-mem-min", str(config.get("tau_mem_min", config["tau_mem"])),
        "--tau-mem-max", str(config.get("tau_mem_max", config["tau_mem"])),
        "--tau-adapt-min", str(config.get("tau_adapt_min", 0.060)),
        "--tau-adapt-max", str(config.get("tau_adapt_max", 0.200)),
        "--adapt-a-min", str(config.get("adapt_a_min", -0.5)),
        "--adapt-a-max", str(config.get("adapt_a_max", 0.5)),
        "--adapt-b-min", str(config.get("adapt_b_min", 0.05)),
        "--adapt-b-max", str(config.get("adapt_b_max", 0.50)),
        "--input-delay-ms",
        *[str(value) for value in config.get("input_delay_ms", [0.0])],
        "--input-channel-group-size", str(config["input_channel_group_size"]),
        "--train-channel-jitter-std", str(config["train_channel_jitter_std"]),
        "--train-jitter-copies", str(config["train_jitter_copies"]),
        "--train-time-scale-min", str(config.get("train_time_scale_min", 1.0)),
        "--train-time-scale-max", str(config.get("train_time_scale_max", 1.0)),
        "--input-branches", str(config.get("input_branches", 1)),
        *(["--deterministic-delay-bank"] if config.get("deterministic_delay_bank", False) else []),
        "--layer2-raw-input-scale", str(config.get("layer2_raw_input_scale", 0.25)),
        "--layer2-connectivity", str(config.get("layer2_connectivity", 0.20)),
        "--layer2-recurrent-delay-ms",
        *[str(value) for value in config.get("layer2_recurrent_delay_ms", [])],
        "--frequency-bandwidths",
        *[str(value) for value in config.get("frequency_bandwidths", [700])],
        "--branch-tau-syn-ms",
        *[str(value) for value in config.get("branch_tau_syn_ms", [])],
        "--proposal-mode", config.get("proposal_mode", "prior"),
        "--guided-proposal-fraction", str(config.get("guided_proposal_fraction", 0.5)),
        "--proposal-time-windows", str(config.get("proposal_time_windows", 16)),
        "--proposal-ema-alpha", str(config.get("proposal_ema_alpha", 0.25)),
        "--proposal-uniform-floor", str(config.get("proposal_uniform_floor", 0.20)),
        "--proposal-fit-ratio", str(config.get("proposal_fit_ratio", 0.40)),
        "--proposal-oof-folds", str(config.get("proposal_oof_folds", 5)),
        "--three-factor-mix", str(config.get("three_factor_mix", 0.0)),
        "--eligibility-tau", str(config.get("eligibility_tau", 0.050)),
        "--membrane-surrogate-width", str(config.get("membrane_surrogate_width", 0.25)),
        "--delay-mutation-control", config.get("delay_mutation_control", "none"),
        "--delay-direction-oof-folds", str(config.get("delay_direction_oof_folds", 5)),
        "--delay-surrogate-gamma", str(config.get("delay_surrogate_gamma", 0.3)),
        "--delay-variant-sample-size", str(config.get("delay_variant_sample_size", 1024)),
        "--delay-mutation-min-mismatch-rate", str(config.get("delay_mutation_min_mismatch_rate", 0.70)),
        "--delay-mutation-min-informative-rounds", str(config.get("delay_mutation_min_informative_rounds", 3)),
        "--threshold", "1.0",
        "--validation-speakers", "2",
        "--batch-size", str(config.get("batch_size", 64)),
        "--num-workers", "8",
        "--seed", str(seed),
        "--device", "cuda",
        "--zo-sample-size", str(config["zo_sample_size"]),
        "--zo-calibration-ratio", "0.20",
        "--zo-sparsity-lambda", str(config["zo_sparsity_lambda"]),
        "--zo-min-keep", str(config["zo_min_keep"]),
        "--compact-loss-tolerance", "0.002",
        "--resume",
    ]
    if args.validation_only:
        command.append("--skip-test")
    print(
        f"task={args.task_id} config={config['config_id']} seed={seed}\n"
        f"config={json.dumps(config, sort_keys=True)}",
        flush=True,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
