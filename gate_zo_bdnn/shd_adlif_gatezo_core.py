"""Frozen current-based adaptive-LIF backend for SHD GateZO experiments.

The structural-growth code expects a small dynamics backend.  This module
keeps that interface while adding a per-neuron slow adaptation current.  All
hidden parameters are sampled once; there is no BPTT or surrogate gradient.
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
    from . import train_shd_bdnn_rsnn_pure as lif_core
except ImportError:
    import train_shd_bdnn_rsnn_pure as lif_core


@dataclass(frozen=True)
class DynamicsConfig:
    input_size: int = 700
    dt: float = 0.0005
    tau_syn: float = 0.010
    tau_mem: float = 0.020
    threshold: float = 1.0
    # Conservative frozen-feature ranges for the slow current.  They are kept
    # in the checkpoint so every run is exactly reproducible and resumable.
    tau_adapt_min: float = 0.060
    tau_adapt_max: float = 0.200
    adapt_a_min: float = -0.5
    adapt_a_max: float = 0.5
    adapt_b_min: float = 0.05
    adapt_b_max: float = 0.50

    def __post_init__(self):
        if self.tau_adapt_min <= 0 or self.tau_adapt_max < self.tau_adapt_min:
            raise ValueError("invalid adaptation time-constant range")
        if self.adapt_a_max < self.adapt_a_min:
            raise ValueError("invalid subthreshold-adaptation range")
        if self.adapt_b_min < 0 or self.adapt_b_max < self.adapt_b_min:
            raise ValueError("invalid spike-triggered-adaptation range")


@dataclass
class RandomRSNNState:
    config: DynamicsConfig
    input_weight: Tensor
    recurrent_weight: Tensor
    bias: Tensor
    tau_adapt: Tensor
    adapt_a: Tensor
    adapt_b: Tensor

    @classmethod
    def empty(cls, config: DynamicsConfig) -> "RandomRSNNState":
        empty = torch.empty(0, dtype=torch.float32)
        return cls(
            config=config,
            input_weight=torch.empty(0, config.input_size, dtype=torch.float32),
            recurrent_weight=torch.empty(0, 0, dtype=torch.float32),
            bias=empty.clone(),
            tau_adapt=empty.clone(),
            adapt_a=empty.clone(),
            adapt_b=empty.clone(),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.input_weight.shape[0])

    def select_from_pool(
        self, old_hidden: int, selected_relative: Sequence[int]
    ) -> "RandomRSNNState":
        selected_relative = np.asarray(selected_relative, dtype=np.int64)
        keep = np.concatenate(
            [np.arange(old_hidden, dtype=np.int64), old_hidden + selected_relative]
        )
        keep_tensor = torch.from_numpy(keep)
        return RandomRSNNState(
            config=self.config,
            input_weight=self.input_weight[keep_tensor].clone(),
            recurrent_weight=self.recurrent_weight[
                keep_tensor[:, None], keep_tensor[None, :]
            ].clone(),
            bias=self.bias[keep_tensor].clone(),
            tau_adapt=self.tau_adapt[keep_tensor].clone(),
            adapt_a=self.adapt_a[keep_tensor].clone(),
            adapt_b=self.adapt_b[keep_tensor].clone(),
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
        }


def load_state(saved) -> RandomRSNNState:
    return RandomRSNNState(
        config=DynamicsConfig(**saved["config"]),
        input_weight=saved["input_weight"].float(),
        recurrent_weight=saved["recurrent_weight"].float(),
        bias=saved["bias"].float(),
        tau_adapt=saved["tau_adapt"].float(),
        adapt_a=saved["adapt_a"].float(),
        adapt_b=saved["adapt_b"].float(),
    )


def propose_individual_pool(
    state: RandomRSNNState,
    pool_size: int,
    input_scales: Sequence[float],
    self_radii: Sequence[float],
    old_to_new_scale: float,
    bias_scale: float,
    rng: np.random.RandomState,
) -> Tuple[RandomRSNNState, list]:
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

    # Log-uniform time constants cover the slow range without concentrating
    # candidates at its high end.  a is subthreshold coupling and b is the
    # spike-triggered increment of the adaptation current.
    tau_adapt = np.exp(
        rng.uniform(
            math.log(state.config.tau_adapt_min),
            math.log(state.config.tau_adapt_max),
            size=pool_size,
        )
    ).astype(np.float32)
    adapt_a = rng.uniform(
        state.config.adapt_a_min, state.config.adapt_a_max, size=pool_size
    ).astype(np.float32)
    adapt_b = rng.uniform(
        state.config.adapt_b_min, state.config.adapt_b_max, size=pool_size
    ).astype(np.float32)

    proposed = RandomRSNNState(
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
        tau_adapt=torch.cat([state.tau_adapt, torch.from_numpy(tau_adapt)]),
        adapt_a=torch.cat([state.adapt_a, torch.from_numpy(adapt_a)]),
        adapt_b=torch.cat([state.adapt_b, torch.from_numpy(adapt_b)]),
    )
    metadata = [
        {
            "candidate": int(index),
            "input_scale": float(scales[index]),
            "self_radius": float(radii[index] * signs[index]),
            "tau_adapt": float(tau_adapt[index]),
            "adapt_a": float(adapt_a[index]),
            "adapt_b": float(adapt_b[index]),
        }
        for index in range(pool_size)
    ]
    return proposed, metadata


recording_steps = lif_core.recording_steps
one_hot = lif_core.one_hot
selected_traces = lif_core.selected_traces


@torch.inference_mode()
def extract_readout_traces(
    loader: DataLoader,
    state: RandomRSNNState,
    device: torch.device,
    readout_bins: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return leaky spike traces, labels, and rates for frozen adLIF neurons."""
    hidden = state.hidden_size
    win = state.input_weight.to(device)
    wrec = state.recurrent_weight.to(device)
    bias = state.bias.to(device)
    tau_adapt = state.tau_adapt.to(device)
    adapt_a = state.adapt_a.to(device)
    adapt_b = state.adapt_b.to(device)
    syn_decay = math.exp(-state.config.dt / state.config.tau_syn)
    mem_decay = math.exp(-state.config.dt / state.config.tau_mem)
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
                current = inputs[:, step] @ win.t() + spike @ wrec.t() + bias
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
