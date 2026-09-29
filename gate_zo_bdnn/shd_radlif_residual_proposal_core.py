"""Supervised residual-adaptive stochastic proposals for frozen RadLIF.

The proposal distribution is updated only after an accepted growth round.  It
changes where the *next* random candidates look; generated parameters remain
frozen and the ordinary GateZO mask/growth gate remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    from . import shd_radlif_advanced_gatezo_core as base
    from .shd_gatezo_support import apply_sequence_readout, solve_sequence_ridge
except ImportError:
    import shd_radlif_advanced_gatezo_core as base
    from shd_gatezo_support import apply_sequence_readout, solve_sequence_ridge


@dataclass(frozen=True)
class DynamicsConfig(base.DynamicsConfig):
    proposal_mode: str = "prior"
    guided_proposal_fraction: float = 0.5
    proposal_time_windows: int = 16
    proposal_ema_alpha: float = 0.25
    proposal_uniform_floor: float = 0.20

    def __post_init__(self):
        super().__post_init__()
        if self.proposal_mode not in {"prior", "residual", "shuffled", "activity"}:
            raise ValueError("invalid proposal_mode")
        if not 0.0 <= self.guided_proposal_fraction <= 1.0:
            raise ValueError("guided_proposal_fraction must be in [0,1]")
        if self.proposal_time_windows < 1:
            raise ValueError("proposal_time_windows must be positive")
        if not 0.0 < self.proposal_ema_alpha <= 1.0:
            raise ValueError("proposal_ema_alpha must be in (0,1]")
        if not 0.0 <= self.proposal_uniform_floor <= 1.0:
            raise ValueError("proposal_uniform_floor must be in [0,1]")


@dataclass
class RandomRSNNState(base.RandomRSNNState):
    proposal_histogram: Tensor
    proposal_updates: int = 0

    @classmethod
    def empty(cls, config: DynamicsConfig) -> "RandomRSNNState":
        raw = base.RandomRSNNState.empty(config)
        histogram = torch.full(
            (config.proposal_time_windows, config.input_size),
            1.0 / (config.proposal_time_windows * config.input_size),
            dtype=torch.float64,
        )
        return _wrap(raw, histogram, 0)

    def select_from_pool(self, old_hidden, selected_relative):
        raw = base.RandomRSNNState.select_from_pool(self, old_hidden, selected_relative)
        return _wrap(raw, self.proposal_histogram.clone(), self.proposal_updates)

    def checkpoint(self) -> Dict:
        saved = base.RandomRSNNState.checkpoint(self)
        saved["config"] = asdict(self.config)
        saved["proposal_histogram"] = self.proposal_histogram
        saved["proposal_updates"] = int(self.proposal_updates)
        return saved


def _wrap(raw, histogram, updates):
    return RandomRSNNState(
        config=raw.config, input_weight=raw.input_weight,
        recurrent_weight=raw.recurrent_weight, bias=raw.bias,
        tau_adapt=raw.tau_adapt, adapt_a=raw.adapt_a, adapt_b=raw.adapt_b,
        tau_syn_values=raw.tau_syn_values, tau_mem_values=raw.tau_mem_values,
        branch_delay_steps=raw.branch_delay_steps,
        branch_band_start=raw.branch_band_start,
        branch_band_stop=raw.branch_band_stop,
        branch_tau_syn_values=raw.branch_tau_syn_values,
        proposal_histogram=histogram, proposal_updates=int(updates),
    )


def load_state(saved) -> RandomRSNNState:
    values = dict(saved["config"])
    for key in ("input_delay_ms", "frequency_bandwidths", "branch_tau_syn_ms"):
        values[key] = tuple(values.get(key, ()))
    config = DynamicsConfig(**values)
    advanced_saved = dict(saved)
    advanced_saved["config"] = {
        key: value for key, value in values.items()
        if key in base.DynamicsConfig.__dataclass_fields__
    }
    raw = base.load_state(advanced_saved)
    raw.config = config
    histogram = saved.get("proposal_histogram")
    if histogram is None:
        histogram = RandomRSNNState.empty(config).proposal_histogram
    return _wrap(raw, histogram.double(), saved.get("proposal_updates", 0))


def propose_individual_pool(state, pool_size, input_scales, self_radii,
                            old_to_new_scale, bias_scale, rng):
    raw, metadata = base.propose_individual_pool(
        state, pool_size, input_scales, self_radii, old_to_new_scale, bias_scale, rng
    )
    proposed = _wrap(raw, state.proposal_histogram.clone(), state.proposal_updates)
    # All arms are bit-identical until the first accepted round supplies evidence.
    if state.config.proposal_mode == "prior" or state.proposal_updates == 0:
        return proposed, metadata

    guided = int(round(pool_size * state.config.guided_proposal_fraction))
    first = pool_size - guided
    old = state.hidden_size
    cdf = np.cumsum(state.proposal_histogram.cpu().numpy().reshape(-1))
    cdf[-1] = 1.0
    available = np.asarray([x for x in state.config.input_delay_steps if x > 0])
    target_ms = np.asarray([10, 25, 50, 100, 200], dtype=np.float64)
    target_steps = target_ms / (state.config.dt * 1000.0)
    for candidate in range(first, pool_size):
        for branch in range(1, state.config.input_branches):
            flat = int(np.searchsorted(cdf, rng.random_sample(), side="right"))
            time_bin, center = divmod(flat, state.config.input_size)
            row = old + candidate
            start = int(proposed.branch_band_start[row, branch])
            stop = int(proposed.branch_band_stop[row, branch])
            width = stop - start
            new_start = min(max(0, center - width // 2), state.config.input_size - width)
            values = proposed.input_weight[row, branch, start:stop].clone()
            proposed.input_weight[row, branch].zero_()
            proposed.input_weight[row, branch, new_start:new_start + width] = values
            proposed.branch_band_start[row, branch] = new_start
            proposed.branch_band_stop[row, branch] = new_start + width
            target = target_steps[min(len(target_steps) - 1,
                                      time_bin * len(target_steps) // state.config.proposal_time_windows)]
            proposed.branch_delay_steps[row, branch] = int(
                available[np.argmin(np.abs(available - target))]
            )
            metadata[candidate]["branch_bands"][branch] = [new_start, new_start + width]
            metadata[candidate]["branch_delay_ms"][branch] = float(
                proposed.branch_delay_steps[row, branch] * state.config.dt * 1000.0
            )
        metadata[candidate]["proposal_source"] = state.config.proposal_mode
    for candidate in range(first):
        metadata[candidate]["proposal_source"] = "fixed_prior"
    return proposed, metadata


def proposal_partitions(labels, proposal_ratio, calibration_ratio, seed):
    """Disjoint D (proposal), GateZO-fit, and GateZO-calibration positions."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    proposal, selection = [], []
    for cls in np.unique(labels):
        positions = np.flatnonzero(labels == cls)
        rng.shuffle(positions)
        count = max(1, int(round(len(positions) * proposal_ratio)))
        proposal.extend(positions[:count]); selection.extend(positions[count:])
    fit, calibration = [], []
    for cls in np.unique(labels):
        positions = np.asarray([p for p in selection if labels[p] == cls])
        rng.shuffle(positions)
        count = max(1, int(round(len(positions) * calibration_ratio)))
        calibration.extend(positions[:count]); fit.extend(positions[count:])
    return tuple(sorted(proposal)), tuple(sorted(fit)), tuple(sorted(calibration))


def _oof_weights(features, labels, positions, folds, ridge, device, seed):
    positions = np.asarray(positions, dtype=np.int64)
    labels_np = labels.cpu().numpy()
    fold_ids = np.empty(len(positions), dtype=np.int64)
    rng = np.random.RandomState(seed)
    for cls in np.unique(labels_np[positions]):
        local = np.flatnonzero(labels_np[positions] == cls); rng.shuffle(local)
        fold_ids[local] = np.arange(len(local)) % folds
    weights = np.zeros(len(positions), dtype=np.float64)
    for fold in range(folds):
        held = positions[fold_ids == fold]; fit = positions[fold_ids != fold]
        beta, _, _, mean, scale = solve_sequence_ridge(
            features[fit], labels[fit], features[held], labels[held], ridge, device
        )
        scores = apply_sequence_readout(features[held], beta, mean, scale)
        probabilities = torch.softmax(scores, 1)
        p_true = probabilities[torch.arange(len(held)), labels[held]].clamp_min(1e-8)
        weights[fold_ids == fold] = torch.clamp(-torch.log(p_true), max=5.0).cpu().numpy()
    return weights


def update_proposal_distribution(state, loader, features, labels, positions,
                                 args, device, round_index):
    if state.config.proposal_mode == "prior":
        return state, {"updated": False, "reason": "fixed_prior"}
    positions = np.asarray(positions, dtype=np.int64)
    labels_np = labels.cpu().numpy()
    if state.config.proposal_mode in {"residual", "shuffled"}:
        weights = _oof_weights(features, labels, positions, args.proposal_oof_folds,
                               args.ridge, device, args.seed + 8803)
        if state.config.proposal_mode == "shuffled":
            rng = np.random.RandomState(args.seed + 9917)
            for cls in np.unique(labels_np[positions]):
                local = np.flatnonzero(labels_np[positions] == cls)
                weights[local] = weights[local][rng.permutation(len(local))]
    else:
        weights = np.ones(len(positions), dtype=np.float64)
    # Equal class mass is part of the supervised controls only.  The activity
    # control deliberately never consults a label after its fixed split exists.
    if state.config.proposal_mode != "activity":
        for cls in np.unique(labels_np[positions]):
            local = np.flatnonzero(labels_np[positions] == cls)
            weights[local] /= max(weights[local].sum(), 1e-12)
    else:
        weights /= max(weights.sum(), 1e-12)
    sample_weight = np.zeros(len(labels_np), dtype=np.float64)
    sample_weight[positions] = weights
    histogram = torch.zeros_like(state.proposal_histogram)
    offset = 0
    for inputs, _ in loader:
        batch = len(inputs); w = torch.from_numpy(sample_weight[offset:offset + batch])
        offset += batch
        if not torch.any(w):
            continue
        bounds = np.linspace(0, inputs.shape[1], state.config.proposal_time_windows + 1).astype(int)
        for window, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
            histogram[window] += (inputs[:, left:right].double().sum(1) * w[:, None]).sum(0)
    histogram += 1e-12
    histogram /= histogram.sum()
    uniform = torch.full_like(histogram, 1.0 / histogram.numel())
    histogram = ((1.0 - state.config.proposal_uniform_floor) * histogram
                 + state.config.proposal_uniform_floor * uniform)
    updated = ((1.0 - state.config.proposal_ema_alpha) * state.proposal_histogram
               + state.config.proposal_ema_alpha * histogram)
    updated /= updated.sum()
    result = _wrap(state, updated, state.proposal_updates + 1)
    entropy = float(-(updated * updated.clamp_min(1e-30).log()).sum())
    return result, {"updated": True, "update_count": result.proposal_updates,
                    "entropy": entropy, "round": int(round_index)}


extract_readout_traces = base.extract_readout_traces
extract_readout_traces_with_eligibility = base.extract_readout_traces_with_eligibility
selected_traces = base.selected_traces
recording_steps = base.recording_steps
one_hot = base.one_hot
