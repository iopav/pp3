"""Deterministic multi-objective search space for structural-ZO SHD runs."""

from __future__ import annotations

import json
import random
from pathlib import Path


FIELDS = (
    "candidate_pool_size",
    "growth_step",
    "zo_min_keep",
    "zo_sparsity_lambda",
    "zo_sample_size",
    "max_hidden",
    "growth_min_delta",
    "diversity_strength",
    "ridge",
    "temporal_windows",
)


def validate_config(config):
    if config["selection_mode"] == "gatezo":
        if not 1 <= config["zo_min_keep"] <= config["growth_step"]:
            raise ValueError("zo_min_keep must be in [1, growth_step]")
    if config["candidate_pool_size"] < config["growth_step"]:
        raise ValueError("candidate pool cannot be smaller than growth_step")
    if (
        config["max_hidden"] != 0
        and config["max_hidden"] < config["zo_min_keep"]
    ):
        raise ValueError("max_hidden cannot be smaller than zo_min_keep")
    return config


def _key(config):
    return tuple(config[field] for field in FIELDS)


def stage1_configs(count=48, seed=20260815):
    """Return curated anchors plus deterministic stratified random configs."""
    anchors = [
        (256, 16, 12, 1e-4, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 16, 12, 1e-4, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 24, 12, 1e-4, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 24, 14, 1e-4, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 24, 16, 1e-4, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 24, 14, 0.0, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 24, 14, 5e-5, 2048, 192, 0.0020, 0.5, 0.5, 12),
        (512, 24, 14, 1e-4, 2048, 176, 0.0020, 0.5, 0.5, 12),
    ]
    configs, seen = [], set()

    def add(values):
        config = {
            "selection_mode": "gatezo",
            **dict(zip(FIELDS, values)),
        }
        validate_config(config)
        key = _key(config)
        if key not in seen:
            seen.add(key)
            configs.append(config)

    for values in anchors:
        add(values)

    rng = random.Random(seed)
    pools = [256, 384, 512, 768]
    growth_steps = [16, 24, 32]
    ratios = [0.50, 0.625, 0.75, 0.875]
    lambdas = [0.0, 2.5e-5, 5e-5, 1e-4, 2e-4]
    samples = [1024, 2048]
    max_hidden = [128, 144, 160, 176, 192, 208]
    deltas = [0.0010, 0.0015, 0.0020, 0.0030]
    diversity = [0.3, 0.5, 0.7]
    ridges = [0.3, 0.5, 0.7]
    windows = [8, 12, 16]
    attempts = 0
    while len(configs) < count:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("could not generate enough unique search configs")
        growth = int(rng.choice(growth_steps))
        ratio = float(rng.choice(ratios))
        keep = max(4, min(growth, int(round(growth * ratio))))
        values = (
            int(rng.choice(pools)), growth, keep, float(rng.choice(lambdas)),
            int(rng.choice(samples)), int(rng.choice(max_hidden)),
            float(rng.choice(deltas)), float(rng.choice(diversity)),
            float(rng.choice(ridges)), int(rng.choice(windows)),
        )
        add(values)

    for index, config in enumerate(configs):
        config["config_id"] = f"cfg{index:03d}"
    return configs


def baseline_configs():
    common = {
        "candidate_pool_size": 256,
        "growth_step": 16,
        "zo_min_keep": 4,
        "zo_sparsity_lambda": 0.0,
        "zo_sample_size": 1024,
        "max_hidden": 512,
        "growth_min_delta": 0.002,
        "diversity_strength": 0.5,
        "ridge": 0.5,
        "temporal_windows": 12,
    }
    return [
        {**common, "config_id": "baseline_fixed16", "selection_mode": "fixed16"},
        {
            **common,
            "config_id": "baseline_adaptive_nozo",
            "selection_mode": "adaptive_nozo",
        },
    ]


def load_configs(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("config file must contain a non-empty JSON list")
    for config in data:
        validate_config(config)
    return data
