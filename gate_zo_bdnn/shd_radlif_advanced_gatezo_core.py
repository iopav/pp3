"""Advanced frozen RadLIF candidates with heterogeneous taus and delay branches.

This backend keeps GateZO and the closed-form ridge readout unchanged.  Each
candidate owns frozen neuron time constants and one or more input branches.
The first branch is full-band; extra branches use randomly selected contiguous
frequency bands and non-zero delays.  No parameter is trained by gradient.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from . import shd_radlif_delay_gatezo_core as delay_core
except ImportError:
    import shd_radlif_delay_gatezo_core as delay_core


@dataclass(frozen=True)
class DynamicsConfig(delay_core.DynamicsConfig):
    heterogeneous_tau_fraction: float = 0.0
    tau_syn_min: float = 0.010
    tau_syn_max: float = 0.010
    tau_mem_min: float = 0.020
    tau_mem_max: float = 0.020
    input_branches: int = 1
    frequency_bandwidths: Tuple[int, ...] = (700,)
    branch_tau_syn_ms: Tuple[float, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.heterogeneous_tau_fraction <= 1.0:
            raise ValueError("heterogeneous_tau_fraction must be in [0, 1]")
        if self.tau_syn_min <= 0.0 or self.tau_syn_max < self.tau_syn_min:
            raise ValueError("invalid synaptic time-constant range")
        if self.tau_mem_min <= 0.0 or self.tau_mem_max < self.tau_mem_min:
            raise ValueError("invalid membrane time-constant range")
        if self.input_branches < 1:
            raise ValueError("input_branches must be positive")
        if not self.frequency_bandwidths or any(
            width < 1 for width in self.frequency_bandwidths
        ):
            raise ValueError("frequency_bandwidths must be positive")
        if self.input_branches > 1 and not any(
            step > 0 for step in self.input_delay_steps
        ):
            raise ValueError("multi-branch dynamics require a non-zero delay")
        if any(value <= 0.0 for value in self.branch_tau_syn_ms):
            raise ValueError("branch_tau_syn_ms values must be positive")


@dataclass
class RandomRSNNState:
    config: DynamicsConfig
    input_weight: Tensor  # [hidden, branch, input]
    recurrent_weight: Tensor
    bias: Tensor
    tau_adapt: Tensor
    adapt_a: Tensor
    adapt_b: Tensor
    tau_syn_values: Tensor
    tau_mem_values: Tensor
    branch_delay_steps: Tensor  # [hidden, branch]
    branch_band_start: Tensor  # [hidden, branch]
    branch_band_stop: Tensor  # [hidden, branch]
    branch_tau_syn_values: Tensor  # seconds, [hidden, branch]

    @classmethod
    def empty(cls, config: DynamicsConfig) -> "RandomRSNNState":
        empty = torch.empty(0, dtype=torch.float32)
        empty_long = torch.empty(0, config.input_branches, dtype=torch.int64)
        return cls(
            config=config,
            input_weight=torch.empty(
                0, config.input_branches, config.input_size, dtype=torch.float32
            ),
            recurrent_weight=torch.empty(0, 0, dtype=torch.float32),
            bias=empty.clone(),
            tau_adapt=empty.clone(),
            adapt_a=empty.clone(),
            adapt_b=empty.clone(),
            tau_syn_values=empty.clone(),
            tau_mem_values=empty.clone(),
            branch_delay_steps=empty_long.clone(),
            branch_band_start=empty_long.clone(),
            branch_band_stop=empty_long.clone(),
            branch_tau_syn_values=torch.empty(
                0, config.input_branches, dtype=torch.float32
            ),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.input_weight.shape[0])

    def select_from_pool(
        self, old_hidden: int, selected_relative: Sequence[int]
    ) -> "RandomRSNNState":
        relative = np.asarray(selected_relative, dtype=np.int64)
        keep = np.concatenate(
            [np.arange(old_hidden, dtype=np.int64), old_hidden + relative]
        )
        index = torch.from_numpy(keep)
        return RandomRSNNState(
            config=self.config,
            input_weight=self.input_weight[index].clone(),
            recurrent_weight=self.recurrent_weight[
                index[:, None], index[None, :]
            ].clone(),
            bias=self.bias[index].clone(),
            tau_adapt=self.tau_adapt[index].clone(),
            adapt_a=self.adapt_a[index].clone(),
            adapt_b=self.adapt_b[index].clone(),
            tau_syn_values=self.tau_syn_values[index].clone(),
            tau_mem_values=self.tau_mem_values[index].clone(),
            branch_delay_steps=self.branch_delay_steps[index].clone(),
            branch_band_start=self.branch_band_start[index].clone(),
            branch_band_stop=self.branch_band_stop[index].clone(),
            branch_tau_syn_values=self.branch_tau_syn_values[index].clone(),
        )

    def checkpoint(self) -> Dict:
        return {
            "config": asdict(self.config),
            "input_weight": self.input_weight,
            "recurrent_weight": self.recurrent_weight,
            "bias": self.bias,
            "tau_adapt": self.tau_adapt,
            "adapt_a": self.adapt_a,
            "adapt_b": self.adapt_b,
            "tau_syn_values": self.tau_syn_values,
            "tau_mem_values": self.tau_mem_values,
            "branch_delay_steps": self.branch_delay_steps,
            "branch_band_start": self.branch_band_start,
            "branch_band_stop": self.branch_band_stop,
            "branch_tau_syn_values": self.branch_tau_syn_values,
        }


def load_state(saved) -> RandomRSNNState:
    config_values = dict(saved["config"])
    config_values["input_delay_ms"] = tuple(
        config_values.get("input_delay_ms", (0.0,))
    )
    config_values["frequency_bandwidths"] = tuple(
        config_values.get("frequency_bandwidths", (700,))
    )
    config_values["branch_tau_syn_ms"] = tuple(
        config_values.get("branch_tau_syn_ms", ())
    )
    config = DynamicsConfig(**config_values)
    hidden = int(saved["input_weight"].shape[0])
    branches = int(saved["input_weight"].shape[1])
    tau_syn_values = saved["tau_syn_values"].float()
    return RandomRSNNState(
        config=config,
        input_weight=saved["input_weight"].float(),
        recurrent_weight=saved["recurrent_weight"].float(),
        bias=saved["bias"].float(),
        tau_adapt=saved["tau_adapt"].float(),
        adapt_a=saved["adapt_a"].float(),
        adapt_b=saved["adapt_b"].float(),
        tau_syn_values=tau_syn_values,
        tau_mem_values=saved["tau_mem_values"].float(),
        branch_delay_steps=saved["branch_delay_steps"].long(),
        branch_band_start=saved["branch_band_start"].long(),
        branch_band_stop=saved["branch_band_stop"].long(),
        branch_tau_syn_values=saved.get(
            "branch_tau_syn_values",
            tau_syn_values[:, None].expand(hidden, branches).clone(),
        ).float(),
    )


def _log_uniform(rng, low: float, high: float, size: int) -> np.ndarray:
    if low == high:
        return np.full(size, low, dtype=np.float32)
    return np.exp(rng.uniform(math.log(low), math.log(high), size=size)).astype(
        np.float32
    )


def propose_individual_pool(
    state: RandomRSNNState,
    pool_size: int,
    input_scales: Sequence[float],
    self_radii: Sequence[float],
    old_to_new_scale: float,
    bias_scale: float,
    rng: np.random.RandomState,
):
    old = state.hidden_size
    total = old + pool_size
    branches = state.config.input_branches
    input_size = state.config.input_size
    scales = rng.choice(np.asarray(input_scales), size=pool_size, replace=True)
    radii = rng.choice(np.asarray(self_radii), size=pool_size, replace=True)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=pool_size, replace=True)

    # Generate the full-band first branch in exactly the original RadLIF order.
    new_weights = np.zeros(
        (pool_size, branches, input_size), dtype=np.float32
    )
    for neuron, scale in enumerate(scales):
        limit = float(scale) / math.sqrt(input_size)
        new_weights[neuron, 0] = rng.uniform(
            -limit, limit, size=input_size
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

    tau_adapt = _log_uniform(
        rng, state.config.tau_adapt_min, state.config.tau_adapt_max, pool_size
    )
    adapt_a = rng.uniform(
        state.config.adapt_a_min, state.config.adapt_a_max, size=pool_size
    ).astype(np.float32)
    adapt_b = rng.uniform(
        state.config.adapt_b_min, state.config.adapt_b_max, size=pool_size
    ).astype(np.float32)
    new_bias = rng.uniform(-bias_scale, bias_scale, size=pool_size).astype(np.float32)

    delays = np.zeros((pool_size, branches), dtype=np.int64)
    starts = np.zeros((pool_size, branches), dtype=np.int64)
    stops = np.full((pool_size, branches), input_size, dtype=np.int64)
    all_delays = np.asarray(state.config.input_delay_steps, dtype=np.int64)
    if branches == 1:
        uniforms = rng.random_sample(pool_size)
        indices = np.minimum(
            (uniforms * len(all_delays)).astype(np.int64), len(all_delays) - 1
        )
        delays[:, 0] = all_delays[indices]
    else:
        nonzero = all_delays[all_delays > 0]
        new_weights[:, 0] /= math.sqrt(branches)
        bandwidths = np.minimum(
            np.asarray(state.config.frequency_bandwidths, dtype=np.int64), input_size
        )
        for neuron, scale in enumerate(scales):
            for branch in range(1, branches):
                delays[neuron, branch] = int(rng.choice(nonzero))
                width = int(rng.choice(bandwidths))
                start = int(rng.randint(0, input_size - width + 1))
                stop = start + width
                starts[neuron, branch] = start
                stops[neuron, branch] = stop
                limit = float(scale) / math.sqrt(width * branches)
                new_weights[neuron, branch, start:stop] = rng.uniform(
                    -limit, limit, size=width
                ).astype(np.float32)

    selector = rng.random_sample(pool_size) < state.config.heterogeneous_tau_fraction
    sampled_syn = _log_uniform(
        rng, state.config.tau_syn_min, state.config.tau_syn_max, pool_size
    )
    sampled_mem = _log_uniform(
        rng, state.config.tau_mem_min, state.config.tau_mem_max, pool_size
    )
    tau_syn = np.where(selector, sampled_syn, state.config.tau_syn).astype(np.float32)
    tau_mem = np.where(selector, sampled_mem, state.config.tau_mem).astype(np.float32)
    if state.config.branch_tau_syn_ms:
        branch_tau_choices = np.asarray(
            state.config.branch_tau_syn_ms, dtype=np.float32
        ) / 1000.0
        branch_tau_syn = rng.choice(
            branch_tau_choices, size=(pool_size, branches), replace=True
        ).astype(np.float32)
    else:
        branch_tau_syn = np.repeat(tau_syn[:, None], branches, axis=1)

    proposed = RandomRSNNState(
        config=state.config,
        input_weight=torch.cat(
            [state.input_weight, torch.from_numpy(new_weights)], dim=0
        ),
        recurrent_weight=recurrent,
        bias=torch.cat([state.bias, torch.from_numpy(new_bias)]),
        tau_adapt=torch.cat([state.tau_adapt, torch.from_numpy(tau_adapt)]),
        adapt_a=torch.cat([state.adapt_a, torch.from_numpy(adapt_a)]),
        adapt_b=torch.cat([state.adapt_b, torch.from_numpy(adapt_b)]),
        tau_syn_values=torch.cat(
            [state.tau_syn_values, torch.from_numpy(tau_syn)]
        ),
        tau_mem_values=torch.cat(
            [state.tau_mem_values, torch.from_numpy(tau_mem)]
        ),
        branch_delay_steps=torch.cat(
            [state.branch_delay_steps, torch.from_numpy(delays)], dim=0
        ),
        branch_band_start=torch.cat(
            [state.branch_band_start, torch.from_numpy(starts)], dim=0
        ),
        branch_band_stop=torch.cat(
            [state.branch_band_stop, torch.from_numpy(stops)], dim=0
        ),
        branch_tau_syn_values=torch.cat(
            [state.branch_tau_syn_values, torch.from_numpy(branch_tau_syn)], dim=0
        ),
    )
    metadata = []
    for index in range(pool_size):
        metadata.append(
            {
                "candidate": int(index),
                "input_scale": float(scales[index]),
                "self_radius": float(radii[index] * signs[index]),
                "tau_adapt": float(tau_adapt[index]),
                "adapt_a": float(adapt_a[index]),
                "adapt_b": float(adapt_b[index]),
                "tau_syn": float(tau_syn[index]),
                "tau_mem": float(tau_mem[index]),
                "heterogeneous_tau": bool(selector[index]),
                "branch_delay_ms": [
                    float(step * state.config.dt * 1000.0)
                    for step in delays[index]
                ],
                "branch_bands": [
                    [int(start), int(stop)]
                    for start, stop in zip(starts[index], stops[index])
                ],
                "branch_tau_syn_ms": [
                    float(value * 1000.0) for value in branch_tau_syn[index]
                ],
            }
        )
    return proposed, metadata


recording_steps = delay_core.recording_steps
one_hot = delay_core.one_hot
selected_traces = delay_core.selected_traces


def _branched_delayed_input_projection(
    inputs: Tensor, weights: Tensor, delays: Tensor
) -> Tensor:
    """Sum full/local branch projections into one current per neuron."""
    batch_size, total_steps, _ = inputs.shape
    hidden, branches, _ = weights.shape
    projected = inputs.new_zeros(batch_size, total_steps, hidden)
    for branch in range(branches):
        branch_delays = delays[:, branch]
        for delay in torch.unique(branch_delays, sorted=True).tolist():
            delay = int(delay)
            if delay >= total_steps:
                continue
            indices = torch.nonzero(
                branch_delays == delay, as_tuple=False
            ).flatten()
            values = inputs[:, : total_steps - delay] @ weights[indices, branch].t()
            projected[:, delay:].index_add_(2, indices, values)
    return projected


def _branched_filtered_input_projection(
    inputs: Tensor,
    weights: Tensor,
    delays: Tensor,
    branch_tau_syn: Tensor,
    dt: float,
) -> Tensor:
    """Apply an independent normalized synaptic filter to every input branch."""
    batch_size, total_steps, _ = inputs.shape
    hidden, branches, _ = weights.shape
    total = inputs.new_zeros(batch_size, total_steps, hidden)
    for branch in range(branches):
        projection = inputs.new_zeros(batch_size, total_steps, hidden)
        branch_delays = delays[:, branch]
        for delay in torch.unique(branch_delays, sorted=True).tolist():
            delay = int(delay)
            if delay >= total_steps:
                continue
            indices = torch.nonzero(
                branch_delays == delay, as_tuple=False
            ).flatten()
            values = inputs[:, : total_steps - delay] @ weights[indices, branch].t()
            projection[:, delay:].index_copy_(2, indices, values)
        decay = torch.exp(-float(dt) / branch_tau_syn[:, branch])
        input_scale = 1.0 - decay
        branch_state = inputs.new_zeros(batch_size, hidden)
        for step in range(total_steps):
            branch_state = decay * branch_state + input_scale * projection[:, step]
            total[:, step].add_(branch_state)
    return total


@torch.inference_mode()
def extract_readout_traces(
    loader: DataLoader,
    state: RandomRSNNState,
    device: torch.device,
    readout_bins: int,
):
    hidden = state.hidden_size
    win = state.input_weight.to(device)
    wrec = state.recurrent_weight.to(device)
    bias = state.bias.to(device)
    tau_adapt = state.tau_adapt.to(device)
    adapt_a = state.adapt_a.to(device)
    adapt_b = state.adapt_b.to(device)
    tau_syn = state.tau_syn_values.to(device)
    tau_mem = state.tau_mem_values.to(device)
    delays = state.branch_delay_steps.to(device)
    branch_tau_syn = state.branch_tau_syn_values.to(device)
    syn_decay = torch.exp(-state.config.dt / tau_syn) if hidden else tau_syn
    mem_decay = torch.exp(-state.config.dt / tau_mem) if hidden else tau_mem
    mem_input_scale = 1.0 - mem_decay
    adapt_decay = torch.exp(-state.config.dt / tau_adapt) if hidden else tau_adapt

    all_features, all_labels = [], []
    spike_total = torch.zeros(hidden, dtype=torch.float64)
    example_steps = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        batch_size, total_steps, _ = inputs.shape
        points = recording_steps(total_steps, readout_bins)
        point_to_bin = {int(step): index for index, step in enumerate(points)}
        traces = inputs.new_zeros(batch_size, readout_bins, hidden)
        if hidden:
            if state.config.branch_tau_syn_ms:
                projected_input = _branched_filtered_input_projection(
                    inputs, win, delays, branch_tau_syn, state.config.dt
                )
            else:
                projected_input = _branched_delayed_input_projection(
                    inputs, win, delays
                )
            syn = inputs.new_zeros(batch_size, hidden)
            mem = inputs.new_zeros(batch_size, hidden)
            spike = inputs.new_zeros(batch_size, hidden)
            adaptation = inputs.new_zeros(batch_size, hidden)
            readout_syn = inputs.new_zeros(batch_size, hidden)
            readout_mem = inputs.new_zeros(batch_size, hidden)
            batch_spike_total = inputs.new_zeros(hidden)
            for step in range(total_steps):
                previous_mem = mem
                adaptation = (
                    adapt_decay * adaptation
                    + (1.0 - adapt_decay) * adapt_a * previous_mem
                    + adapt_b * spike
                )
                current = projected_input[:, step] + spike @ wrec.t() + bias
                syn = syn_decay * syn + current
                mem = mem_decay * mem + mem_input_scale * (syn - adaptation)
                spike = (mem > state.config.threshold).to(inputs.dtype)
                mem = mem * (1.0 - spike)
                readout_syn = syn_decay * readout_syn + spike
                readout_mem = mem_decay * readout_mem + mem_input_scale * readout_syn
                if step in point_to_bin:
                    traces[:, point_to_bin[step], :] = readout_mem
                batch_spike_total += spike.sum(dim=0)
            spike_total += batch_spike_total.double().cpu()
            example_steps += batch_size * total_steps
        all_features.append(traces.cpu())
        all_labels.append(labels.clone())
    rates = spike_total / float(max(1, example_steps))
    return torch.cat(all_features), torch.cat(all_labels), rates


@torch.inference_mode()
def extract_readout_traces_with_eligibility(
    loader: DataLoader,
    state: RandomRSNNState,
    device: torch.device,
    readout_bins: int,
    readout_warmup_bins: int,
    old_hidden: int,
    temporal_windows: int,
    eligibility_tau: float,
    membrane_surrogate_width: float,
):
    """Return ordinary traces plus a local pre/post eligibility per candidate.

    The eligibility is diagnostic structural credit only: no weight, delay, or
    neuron parameter is updated.  It combines absolute afferent drive with a
    triangular near-threshold postsynaptic sensitivity and a causal trace.
    """
    if eligibility_tau <= 0.0 or membrane_surrogate_width <= 0.0:
        raise ValueError("eligibility tau and surrogate width must be positive")
    hidden = state.hidden_size
    candidates = hidden - int(old_hidden)
    if candidates <= 0:
        raise ValueError("state must contain new candidate neurons")
    win = state.input_weight.to(device)
    wrec = state.recurrent_weight.to(device)
    bias = state.bias.to(device)
    tau_adapt = state.tau_adapt.to(device)
    adapt_a = state.adapt_a.to(device)
    adapt_b = state.adapt_b.to(device)
    tau_syn = state.tau_syn_values.to(device)
    tau_mem = state.tau_mem_values.to(device)
    delays = state.branch_delay_steps.to(device)
    branch_tau_syn = state.branch_tau_syn_values.to(device)
    syn_decay = torch.exp(-state.config.dt / tau_syn)
    mem_decay = torch.exp(-state.config.dt / tau_mem)
    mem_input_scale = 1.0 - mem_decay
    adapt_decay = torch.exp(-state.config.dt / tau_adapt)
    eligibility_decay = math.exp(-state.config.dt / float(eligibility_tau))

    all_features, all_labels, all_eligibility = [], [], []
    spike_total = torch.zeros(hidden, dtype=torch.float64)
    example_steps = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        batch_size, total_steps, _ = inputs.shape
        points = recording_steps(total_steps, readout_bins)
        point_to_bin = {int(step): index for index, step in enumerate(points)}
        warmup_steps = int(round(
            total_steps * float(readout_warmup_bins) / float(readout_bins)
        ))
        boundaries = np.linspace(
            warmup_steps, total_steps, temporal_windows + 1, dtype=np.int64
        )
        traces = inputs.new_zeros(batch_size, readout_bins, hidden)
        eligibility_windows = inputs.new_zeros(
            batch_size, temporal_windows, candidates
        )
        window_counts = inputs.new_zeros(temporal_windows)
        if state.config.branch_tau_syn_ms:
            projected_input = _branched_filtered_input_projection(
                inputs, win, delays, branch_tau_syn, state.config.dt
            )
        else:
            projected_input = _branched_delayed_input_projection(inputs, win, delays)
        syn = inputs.new_zeros(batch_size, hidden)
        mem = inputs.new_zeros(batch_size, hidden)
        spike = inputs.new_zeros(batch_size, hidden)
        adaptation = inputs.new_zeros(batch_size, hidden)
        readout_syn = inputs.new_zeros(batch_size, hidden)
        readout_mem = inputs.new_zeros(batch_size, hidden)
        eligibility = inputs.new_zeros(batch_size, candidates)
        batch_spike_total = inputs.new_zeros(hidden)
        window = 0
        for step in range(total_steps):
            active_window = step >= warmup_steps
            while window + 1 < temporal_windows and step >= boundaries[window + 1]:
                window += 1
            previous_mem = mem
            adaptation = (
                adapt_decay * adaptation
                + (1.0 - adapt_decay) * adapt_a * previous_mem
                + adapt_b * spike
            )
            afferent = projected_input[:, step] + spike @ wrec.t()
            current = afferent + bias
            syn = syn_decay * syn + current
            pre_reset_mem = mem_decay * mem + mem_input_scale * (syn - adaptation)
            post = torch.clamp(
                1.0
                - torch.abs(pre_reset_mem[:, old_hidden:] - state.config.threshold)
                / float(membrane_surrogate_width),
                min=0.0,
            )
            pre = torch.abs(afferent[:, old_hidden:])
            eligibility = eligibility_decay * eligibility + pre * post
            if active_window:
                eligibility_windows[:, window].add_(eligibility)
                window_counts[window] += 1.0
            spike = (pre_reset_mem > state.config.threshold).to(inputs.dtype)
            mem = pre_reset_mem * (1.0 - spike)
            readout_syn = syn_decay * readout_syn + spike
            readout_mem = mem_decay * readout_mem + mem_input_scale * readout_syn
            if step in point_to_bin:
                traces[:, point_to_bin[step], :] = readout_mem
            batch_spike_total += spike.sum(dim=0)
        eligibility_windows /= window_counts.clamp_min(1.0)[None, :, None]
        spike_total += batch_spike_total.double().cpu()
        example_steps += batch_size * total_steps
        all_features.append(traces.cpu())
        all_labels.append(labels.clone())
        all_eligibility.append(eligibility_windows.cpu())
    rates = spike_total / float(max(1, example_steps))
    return (
        torch.cat(all_features), torch.cat(all_labels), rates,
        torch.cat(all_eligibility),
    )
