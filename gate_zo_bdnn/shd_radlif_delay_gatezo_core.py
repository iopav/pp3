"""Frozen RadLIF backend with one fixed input delay per hidden neuron.

Weights, recurrent connections, RadLIF adaptation parameters, and delays are
sampled once and remain frozen.  GateZO selects useful neurons and the output
weights are still obtained by closed-form ridge regression.  A delay applies
to the complete input projection of one neuron; recurrent spikes are not
delayed in this first controlled experiment.
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
    from . import shd_adlif_gatezo_core as radlif
except ImportError:
    import shd_adlif_gatezo_core as radlif


@dataclass(frozen=True)
class DynamicsConfig(radlif.DynamicsConfig):
    input_delay_ms: Tuple[float, ...] = (0.0,)

    def __post_init__(self):
        super().__post_init__()
        if not self.input_delay_ms:
            raise ValueError("input_delay_ms cannot be empty")
        if any(value < 0.0 for value in self.input_delay_ms):
            raise ValueError("input delays must be nonnegative")
        for value in self.input_delay_ms:
            seconds = float(value) / 1000.0
            steps = round(seconds / self.dt)
            if not math.isclose(steps * self.dt, seconds, abs_tol=1e-10):
                raise ValueError(
                    f"input delay {value} ms is not aligned to dt={self.dt} s"
                )

    @property
    def input_delay_steps(self) -> Tuple[int, ...]:
        return tuple(
            int(round((float(value) / 1000.0) / self.dt))
            for value in self.input_delay_ms
        )


@dataclass
class RandomRSNNState:
    config: DynamicsConfig
    input_weight: Tensor
    recurrent_weight: Tensor
    bias: Tensor
    tau_adapt: Tensor
    adapt_a: Tensor
    adapt_b: Tensor
    input_delay_steps: Tensor

    @classmethod
    def empty(cls, config: DynamicsConfig) -> "RandomRSNNState":
        base = radlif.RandomRSNNState.empty(config)
        return cls(
            config=config,
            input_weight=base.input_weight,
            recurrent_weight=base.recurrent_weight,
            bias=base.bias,
            tau_adapt=base.tau_adapt,
            adapt_a=base.adapt_a,
            adapt_b=base.adapt_b,
            input_delay_steps=torch.empty(0, dtype=torch.int64),
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
            input_delay_steps=self.input_delay_steps[keep_tensor].clone(),
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
            "input_delay_steps": self.input_delay_steps,
        }


def load_state(saved) -> RandomRSNNState:
    config_values = dict(saved["config"])
    config_values.setdefault("input_delay_ms", (0.0,))
    config = DynamicsConfig(**config_values)
    hidden = int(saved["input_weight"].shape[0])
    return RandomRSNNState(
        config=config,
        input_weight=saved["input_weight"].float(),
        recurrent_weight=saved["recurrent_weight"].float(),
        bias=saved["bias"].float(),
        tau_adapt=saved["tau_adapt"].float(),
        adapt_a=saved["adapt_a"].float(),
        adapt_b=saved["adapt_b"].float(),
        input_delay_steps=saved.get(
            "input_delay_steps", torch.zeros(hidden, dtype=torch.int64)
        ).long(),
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
    """Generate the original RadLIF candidates, then attach one delay each."""
    base_state = radlif.RandomRSNNState(
        config=state.config,
        input_weight=state.input_weight,
        recurrent_weight=state.recurrent_weight,
        bias=state.bias,
        tau_adapt=state.tau_adapt,
        adapt_a=state.adapt_a,
        adapt_b=state.adapt_b,
    )
    proposed, metadata = radlif.propose_individual_pool(
        base_state,
        pool_size,
        input_scales,
        self_radii,
        old_to_new_scale,
        bias_scale,
        rng,
    )

    # Consume exactly one uniform draw per candidate for both delay=0 and
    # multi-delay configurations.  Thus all preceding weight and RadLIF draws
    # are paired across the two configurations.
    choices = np.asarray(state.config.input_delay_steps, dtype=np.int64)
    uniform = rng.random_sample(pool_size)
    choice_indices = np.minimum(
        (uniform * len(choices)).astype(np.int64), len(choices) - 1
    )
    new_delays = choices[choice_indices]
    delays = torch.cat(
        [state.input_delay_steps, torch.from_numpy(new_delays)], dim=0
    )
    for index, delay_steps in enumerate(new_delays):
        metadata[index]["input_delay_steps"] = int(delay_steps)
        metadata[index]["input_delay_ms"] = float(
            delay_steps * state.config.dt * 1000.0
        )

    return RandomRSNNState(
        config=state.config,
        input_weight=proposed.input_weight,
        recurrent_weight=proposed.recurrent_weight,
        bias=proposed.bias,
        tau_adapt=proposed.tau_adapt,
        adapt_a=proposed.adapt_a,
        adapt_b=proposed.adapt_b,
        input_delay_steps=delays,
    ), metadata


recording_steps = radlif.recording_steps
one_hot = radlif.one_hot
selected_traces = radlif.selected_traces


def _delayed_input_projection(inputs: Tensor, win: Tensor, delays: Tensor) -> Tensor:
    """Project a full batch sequence using neuron-level fixed delays."""
    batch_size, total_steps, _ = inputs.shape
    hidden = int(win.shape[0])
    projected = inputs.new_zeros(batch_size, total_steps, hidden)
    for delay in torch.unique(delays, sorted=True).tolist():
        delay = int(delay)
        if delay >= total_steps:
            continue
        indices = torch.nonzero(delays == delay, as_tuple=False).flatten()
        values = inputs[:, : total_steps - delay] @ win[indices].t()
        projected[:, delay:].index_copy_(2, indices, values)
    return projected


@torch.inference_mode()
def extract_readout_traces(
    loader: DataLoader,
    state: RandomRSNNState,
    device: torch.device,
    readout_bins: int,
):
    """Return leaky spike traces, labels, and rates for delayed RadLIF."""
    hidden = state.hidden_size
    win = state.input_weight.to(device)
    wrec = state.recurrent_weight.to(device)
    bias = state.bias.to(device)
    tau_adapt = state.tau_adapt.to(device)
    adapt_a = state.adapt_a.to(device)
    adapt_b = state.adapt_b.to(device)
    delays = state.input_delay_steps.to(device)
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
            projected_input = _delayed_input_projection(inputs, win, delays)
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

