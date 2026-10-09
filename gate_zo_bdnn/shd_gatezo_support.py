"""Self-contained readout, sampling, and candidate-selection helpers for SHD.

This file keeps the RadLIF B/C experiment inside ``gate_zo_bdnn`` instead of
depending on similarly named research scripts at the repository root.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Subset

try:
    from . import train_shd_bdnn_rsnn_pure as core
except ImportError:
    import train_shd_bdnn_rsnn_pure as core


def propose_individual_pool(
    state,
    pool_size: int,
    input_scales,
    self_radii,
    old_to_new_scale: float,
    bias_scale: float,
    rng: np.random.RandomState,
):
    """Local ordinary-LIF proposal used by the shared structural trainer."""
    old = state.hidden_size
    total = old + pool_size
    scales = rng.choice(np.asarray(input_scales), size=pool_size, replace=True)
    radii = rng.choice(np.asarray(self_radii), size=pool_size, replace=True)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=pool_size, replace=True)
    win = np.empty((pool_size, state.config.input_size), dtype=np.float32)
    for index, scale in enumerate(scales):
        limit = float(scale) / math.sqrt(state.config.input_size)
        win[index] = rng.uniform(
            -limit, limit, size=state.config.input_size
        ).astype(np.float32)
    recurrent = torch.zeros(total, total, dtype=torch.float32)
    if old:
        recurrent[:old, :old].copy_(state.recurrent_weight)
        limit = float(old_to_new_scale) / math.sqrt(old)
        recurrent[old:, :old] = torch.from_numpy(
            rng.uniform(-limit, limit, size=(pool_size, old)).astype(np.float32)
        )
    recurrent[old:, old:] = torch.diag(
        torch.from_numpy((radii * signs).astype(np.float32))
    )
    proposed = core.RandomRSNNState(
        config=state.config,
        input_weight=torch.cat([state.input_weight, torch.from_numpy(win)], dim=0),
        recurrent_weight=recurrent,
        bias=torch.cat(
            [
                state.bias,
                torch.from_numpy(
                    rng.uniform(-bias_scale, bias_scale, size=pool_size).astype(
                        np.float32
                    )
                ),
            ]
        ),
    )
    metadata = [
        {
            "candidate": int(index),
            "input_scale": float(scales[index]),
            "self_radius": float(radii[index] * signs[index]),
        }
        for index in range(pool_size)
    ]
    return proposed, metadata


def dataset_labels(dataset) -> np.ndarray:
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "labels"):
        return np.asarray(dataset.dataset.labels)[np.asarray(dataset.indices)]
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return np.asarray([dataset[index][2] for index in range(len(dataset))])


def stratified_fit_calibration_split(labels, ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    fit, calibration = [], []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        rng.shuffle(positions)
        count = max(1, min(len(positions) - 1, int(round(len(positions) * ratio))))
        calibration.extend(positions[:count].tolist())
        fit.extend(positions[count:].tolist())
    return np.asarray(sorted(fit)), np.asarray(sorted(calibration))


def balanced_sample(pool, labels, count: int, rng: np.random.RandomState):
    pool = np.asarray(pool)
    if pool.size == 0 or count == 0:
        return []
    if pool.ndim != 1 or not np.issubdtype(pool.dtype, np.integer):
        raise ValueError('Sampling pool must be a one-dimensional integer index array')
    if count < 0:
        raise ValueError('Sample count must be nonnegative')
    labels = np.asarray(labels)
    classes = np.unique(labels[pool])
    base, extra = divmod(min(int(count), len(pool)), len(classes))
    chosen = []
    for offset, label in enumerate(classes):
        candidates = pool[labels[pool] == label]
        take = min(len(candidates), base + int(offset < extra))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(chosen) < min(int(count), len(pool)):
        remaining = np.setdiff1d(pool, np.asarray(chosen), assume_unique=False)
        take = min(len(remaining), min(int(count), len(pool)) - len(chosen))
        chosen.extend(rng.choice(remaining, size=take, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def smoke_training_indices(labels, indices, limit):
    """Keep four originals per class for disjoint proposal/fit/calibration."""
    labels = np.asarray(labels)
    indices = np.asarray(indices, dtype=np.int64)
    pools = [indices[labels[indices] == label] for label in np.unique(labels[indices])]
    if not pools or any(len(pool) < 4 for pool in pools):
        raise ValueError('GateZO SHD smoke requires at least four training originals per class after speaker splitting')
    count = min(len(indices), max(int(limit), 4 * len(pools)))
    chosen = []
    for offset in range(max(map(len, pools))):
        for pool in pools:
            if offset < len(pool):
                chosen.append(int(pool[offset]))
                if len(chosen) == count:
                    return np.asarray(chosen, dtype=np.int64)


def temporal_window_features(
    traces: Tensor,
    warmup_bins: int,
    temporal_windows: int,
    pooling: str = "window_mean",
    attention_temperature: float = 1.0,
    attention_activity_mix: float = 0.5,
) -> Tensor:
    """Return one sequence-level feature matrix per utterance."""
    selected = traces[:, warmup_bins:, :]
    if selected.shape[1] < 1:
        raise ValueError("at least one readout bin must remain after warmup")
    if pooling == "cumulative_sum":
        # Linear output weights commute with summation, so this is equivalent
        # to accumulating the non-spiking class readout over the utterance.
        return selected.sum(dim=1)
    if pooling == "parameter_free_attention_sum":
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if not 0.0 <= attention_activity_mix <= 1.0:
            raise ValueError("attention_activity_mix must be in [0, 1]")
        previous = torch.cat([selected[:, :1, :], selected[:, :-1, :]], dim=1)
        importance = (
            attention_activity_mix * selected.abs()
            + (1.0 - attention_activity_mix) * (selected - previous).abs()
        )
        importance = importance - importance.mean(dim=1, keepdim=True)
        importance = importance / importance.std(
            dim=1, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        attention = torch.softmax(importance / attention_temperature, dim=1)
        return torch.sum(attention * selected, dim=1)
    if pooling != "window_mean":
        raise ValueError(f"unsupported temporal pooling: {pooling}")
    if temporal_windows < 1 or temporal_windows > selected.shape[1]:
        raise ValueError("temporal_windows must fit after warmup")
    boundaries = np.linspace(
        0, selected.shape[1], temporal_windows + 1, dtype=np.int64
    )
    pooled = []
    for index in range(temporal_windows):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        if stop <= start:
            raise ValueError("empty temporal window")
        pooled.append(selected[:, start:stop, :].mean(dim=1))
    return torch.cat(pooled, dim=1)


def solve_sequence_ridge(
    train_features: Tensor,
    train_labels: Tensor,
    eval_features: Tensor,
    eval_labels: Tensor,
    ridge: float,
    solve_device: torch.device,
) -> Tuple[Tensor, Dict[str, float], Dict[str, float], Tensor, Tensor]:
    train = train_features.double()
    evaluate = eval_features.double()
    if train.shape[1]:
        mean = train.mean(dim=0)
        scale = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        mean = torch.empty(0, dtype=torch.float64)
        scale = torch.empty(0, dtype=torch.float64)
    train = (train - mean) / scale
    evaluate = (evaluate - mean) / scale
    targets = core.one_hot(train_labels, num_classes=20)
    rows = max(1, train.shape[0])
    if train.shape[1] > train.shape[0]:
        # Standardized feature columns have zero mean, so the unpenalized
        # intercept separates exactly and the smaller sample-space system can
        # be solved without constructing a potentially huge feature Gram.
        intercept = targets.mean(0)
        centered = targets - intercept
        x = train.to(solve_device)
        y = centered.to(solve_device)
        kernel = (x @ x.t()) / float(rows)
        kernel.diagonal().add_(float(ridge))
        try:
            dual = torch.linalg.solve(kernel, y / math.sqrt(rows))
        except torch.linalg.LinAlgError:
            dual = torch.linalg.lstsq(kernel, y / math.sqrt(rows)).solution
        coefficients = (x.t() @ dual / math.sqrt(rows)).cpu()
        beta = torch.cat([intercept[None], coefficients], dim=0)
    else:
        train_design_device = torch.cat([
            torch.ones(train.shape[0], 1, dtype=torch.float64), train
        ], dim=1).to(solve_device)
        y = targets.to(solve_device)
        gram = (train_design_device.t() @ train_design_device) / float(rows)
        rhs = (train_design_device.t() @ y) / float(rows)
        penalty = torch.eye(
            gram.shape[0], dtype=torch.float64, device=solve_device
        ) * float(ridge)
        penalty[0, 0] = 0.0
        try:
            beta = torch.linalg.solve(gram + penalty, rhs)
        except torch.linalg.LinAlgError:
            beta = torch.linalg.lstsq(gram + penalty, rhs).solution
        beta = beta.cpu()
    train_design = torch.cat(
        [torch.ones(train.shape[0], 1, dtype=torch.float64), train], dim=1
    )
    eval_design = torch.cat(
        [torch.ones(evaluate.shape[0], 1, dtype=torch.float64), evaluate], dim=1
    )
    train_metrics = core.classification_metrics(train_design @ beta, train_labels)
    eval_metrics = core.classification_metrics(eval_design @ beta, eval_labels)
    return beta, train_metrics, eval_metrics, mean, scale


def apply_sequence_readout(features: Tensor, beta: Tensor, mean: Tensor, scale: Tensor):
    normalized = (features.double() - mean) / scale
    design = torch.cat(
        [torch.ones(normalized.shape[0], 1, dtype=torch.float64), normalized], dim=1
    )
    return design @ beta


def profiled_ridge_metrics(
    fit_features,
    fit_labels,
    eval_features,
    eval_labels,
    ridge: float,
    device,
):
    """Ridge with an exact dual solve when samples are fewer than features."""
    fit = fit_features.double()
    evaluate = eval_features.double()
    mean = fit.mean(0)
    scale = fit.std(0, unbiased=False).clamp_min(1e-6)
    x = (fit - mean) / scale
    z = (evaluate - mean) / scale
    targets = core.one_hot(fit_labels, 20)
    intercept = targets.mean(0)
    centered = targets - intercept
    rows = max(1, x.shape[0])
    x_device = x.to(device)
    y_device = centered.to(device)
    if x.shape[1] > x.shape[0]:
        kernel = (x_device @ x_device.t()) / float(rows)
        kernel.diagonal().add_(float(ridge))
        dual = torch.linalg.solve(kernel, y_device / math.sqrt(rows))
        coefficients = (x_device.t() @ dual / math.sqrt(rows)).cpu()
    else:
        gram = (x_device.t() @ x_device) / float(rows)
        gram.diagonal().add_(float(ridge))
        rhs = (x_device.t() @ y_device) / float(rows)
        coefficients = torch.linalg.solve(gram, rhs).cpu()
    fit_scores = x @ coefficients + intercept
    eval_scores = z @ coefficients + intercept
    return (
        core.classification_metrics(fit_scores, fit_labels),
        core.classification_metrics(eval_scores, eval_labels),
    )


def _pool_window_tensor(traces, warmup_bins: int, windows: int):
    selected = traces[:, warmup_bins:, :]
    boundaries = np.linspace(0, selected.shape[1], windows + 1, dtype=np.int64)
    return torch.stack(
        [
            selected[:, int(boundaries[i]) : int(boundaries[i + 1]), :].mean(1)
            for i in range(windows)
        ],
        dim=1,
    )


def select_residual_diverse_neurons(
    current_train_features,
    current_beta,
    pool_train_traces,
    train_labels,
    old_hidden: int,
    candidate_rates,
    args,
    three_factor_eligibility=None,
):
    """One-pass group relevance followed by greedy diversity suppression."""
    candidate_traces = pool_train_traces[:, :, old_hidden:]
    if getattr(args, "temporal_pooling", "window_mean") == "window_mean":
        candidate_tensor = _pool_window_tensor(
            candidate_traces, args.readout_warmup_bins, args.temporal_windows
        ).double()
    else:
        candidate_features = temporal_window_features(
            candidate_traces,
            args.readout_warmup_bins,
            args.temporal_windows,
            pooling=args.temporal_pooling,
            attention_temperature=args.attention_temperature,
            attention_activity_mix=args.attention_activity_mix,
        )
        candidate_tensor = candidate_features[:, None, :].double()
    mean = candidate_tensor.mean(dim=0, keepdim=True)
    scale = candidate_tensor.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    normalized = (candidate_tensor - mean) / scale
    current_scores = apply_sequence_readout(
        current_train_features,
        current_beta,
        args.current_feature_mean,
        args.current_feature_scale,
    )
    residual = core.one_hot(train_labels, 20) - current_scores
    base_relevance = torch.einsum(
        "nwp,nc->pwc", normalized, residual
    ).pow(2).sum((1, 2))
    base_relevance /= float(max(1, normalized.shape[0])) ** 2
    mix = float(getattr(args, "three_factor_mix", 0.0))
    three_factor_relevance = None
    if mix > 0.0:
        if three_factor_eligibility is None:
            raise ValueError("three-factor eligibility is required when mix > 0")
        eligibility = three_factor_eligibility.double()
        if eligibility.shape != normalized.shape:
            raise ValueError("three-factor eligibility shape does not match candidates")
        eligibility = (
            eligibility - eligibility.mean(0, keepdim=True)
        ) / eligibility.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        three_factor_relevance = torch.einsum(
            "nwp,nc->pwc", eligibility, residual
        ).pow(2).sum((1, 2))
        three_factor_relevance /= float(max(1, eligibility.shape[0])) ** 2
        base_unit = base_relevance / base_relevance.mean().clamp_min(1e-12)
        three_unit = three_factor_relevance / three_factor_relevance.mean().clamp_min(1e-12)
        relevance = (1.0 - mix) * base_unit + mix * three_unit
    else:
        relevance = base_relevance
    flat = normalized.permute(2, 0, 1).reshape(normalized.shape[2], -1)
    flat /= torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    similarity = torch.abs(flat @ flat.t())
    valid = (candidate_rates >= args.candidate_min_rate) & (
        candidate_rates <= args.candidate_max_rate
    )
    selected: List[int] = []
    records: List[Dict] = []
    max_similarity = torch.zeros_like(relevance)
    available = valid.clone()
    for slot in range(min(args.growth_step, int(valid.sum()))):
        score = relevance * (1.0 - args.diversity_strength * max_similarity.pow(2))
        score[~available] = -torch.inf
        chosen = int(torch.argmax(score))
        chosen_score = float(score[chosen])
        if not math.isfinite(chosen_score) or chosen_score <= args.min_candidate_score:
            break
        selected.append(chosen)
        records.append(
            {
                "slot": slot,
                "candidate": chosen,
                "score": chosen_score,
                "relevance": float(relevance[chosen]),
                "base_relevance": float(base_relevance[chosen]),
                "three_factor_relevance": (
                    float(three_factor_relevance[chosen])
                    if three_factor_relevance is not None else None
                ),
                "max_similarity": float(max_similarity[chosen]),
                "rate": float(candidate_rates[chosen]),
            }
        )
        available[chosen] = False
        max_similarity = torch.maximum(max_similarity, similarity[chosen])
    return selected, records
