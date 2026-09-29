"""Pure-screening BDNN-style growing RSNN for the SHD data set.

There is no gradient training, BPTT, epoch loop, learning rate, or ZO in this
program.  Every hidden parameter is randomly configured once.  Growth uses:

    random candidate pool -> residual screening -> closed-form ridge readout
    -> validation commit/rollback.

To preserve the nested feature space required by residual screening, recurrent
connections are block-lower-triangular in neuron birth order: old neurons may
drive new neurons, while a candidate never feeds back into committed neurons.

The readout remains temporal.  Hidden spikes are passed through fixed leaky
synapse/membrane traces.  A shared linear readout is solved on sampled time
points in closed form, and inference uses max-over-time over those readout
traces.  Thus max-over-time inference is retained without iterative training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


class SHDEvents(Dataset):
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError("SHD file not found: {}".format(self.path))
        self._file: Optional[h5py.File] = None
        with h5py.File(self.path, "r") as handle:
            self.labels = np.asarray(handle["labels"], dtype=np.int64)
            if "extra/speaker" in handle:
                raw_speakers = handle["extra/speaker"][:]
                self.speakers = np.asarray(
                    [
                        item.decode("utf-8")
                        if isinstance(item, (bytes, np.bytes_))
                        else str(item)
                        for item in raw_speakers
                    ]
                )
                if self.speakers.shape[0] != self.labels.shape[0]:
                    raise ValueError("speaker metadata length does not match labels")
            else:
                self.speakers = None
            if "extra/keys" in handle:
                self.keys = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in handle["extra/keys"][:]
                ]
            else:
                self.keys = [str(index) for index in range(20)]

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        handle = self._handle()
        return (
            np.asarray(handle["spikes/times"][index], dtype=np.float32),
            np.asarray(handle["spikes/units"][index], dtype=np.int64),
            int(self.labels[index]),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


class SHDBinner:
    def __init__(
        self,
        dt: float,
        max_time: float,
        channels: int = 700,
        channel_group_size: int = 1,
    ):
        if dt <= 0.0 or max_time <= 0.0:
            raise ValueError("dt and max_time must be positive")
        if channels <= 0 or channel_group_size <= 0:
            raise ValueError("channels and channel_group_size must be positive")
        self.dt = float(dt)
        self.steps = int(math.ceil(max_time / dt))
        self.source_channels = int(channels)
        self.channel_group_size = int(channel_group_size)
        self.channels = int(math.ceil(self.source_channels / self.channel_group_size))

    def __call__(self, batch):
        frames = torch.zeros(
            len(batch), self.steps, self.channels, dtype=torch.float32
        )
        labels = torch.empty(len(batch), dtype=torch.long)
        flat = frames.view(-1)
        sample_stride = self.steps * self.channels
        for sample, (times, units, label) in enumerate(batch):
            time_index = np.floor(times / self.dt).astype(np.int64)
            valid = (
                (time_index >= 0)
                & (time_index < self.steps)
                & (units >= 0)
                & (units < self.source_channels)
            )
            if np.any(valid):
                mapped_units = units[valid] // self.channel_group_size
                indices = (
                    sample * sample_stride
                    + time_index[valid] * self.channels
                    + mapped_units
                ).astype(np.int64, copy=False)
                index_tensor = torch.from_numpy(indices)
                flat.index_fill_(0, index_tensor, 1.0)
            labels[sample] = int(label)
        return frames, labels


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    rng = np.random.RandomState(seed)
    train_parts, val_parts = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = min(max(1, int(round(indices.size * val_ratio))), indices.size - 1)
        val_parts.append(indices[:count])
        train_parts.append(indices[count:])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


@dataclass(frozen=True)
class DynamicsConfig:
    input_size: int = 700
    dt: float = 0.005
    tau_syn: float = 0.010
    tau_mem: float = 0.020
    threshold: float = 1.0


@dataclass
class RandomRSNNState:
    config: DynamicsConfig
    input_weight: Tensor
    recurrent_weight: Tensor
    bias: Tensor

    @classmethod
    def empty(cls, config: DynamicsConfig) -> "RandomRSNNState":
        return cls(
            config=config,
            input_weight=torch.empty(0, config.input_size, dtype=torch.float32),
            recurrent_weight=torch.empty(0, 0, dtype=torch.float32),
            bias=torch.empty(0, dtype=torch.float32),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.input_weight.shape[0])

    def propose_pool(
        self,
        pool_size: int,
        lambdas: Sequence[float],
        rng: np.random.RandomState,
        input_scale: float,
        recurrent_scale: float,
        self_recurrent_scale: float,
        bias_scale: float,
    ) -> Tuple["RandomRSNNState", np.ndarray]:
        old = self.hidden_size
        pool_size = int(pool_size)
        if pool_size <= 0 or not lambdas:
            raise ValueError("pool_size and lambdas must be non-empty")
        scales = np.resize(np.asarray(lambdas, dtype=np.float64), pool_size)
        rng.shuffle(scales)

        win_new = np.empty((pool_size, self.config.input_size), dtype=np.float32)
        old_to_new = np.zeros((pool_size, old), dtype=np.float32)
        self_weights = np.empty(pool_size, dtype=np.float32)
        for index, scale in enumerate(scales):
            input_limit = float(scale) * input_scale / math.sqrt(self.config.input_size)
            win_new[index] = rng.uniform(
                -input_limit, input_limit, size=self.config.input_size
            )
            if old:
                recurrent_limit = (
                    float(scale) * recurrent_scale / math.sqrt(old)
                )
                old_to_new[index] = rng.uniform(
                    -recurrent_limit, recurrent_limit, size=old
                )
            self_weights[index] = rng.uniform(
                -self_recurrent_scale, self_recurrent_scale
            )

        new_hidden = old + pool_size
        recurrent = torch.zeros(new_hidden, new_hidden, dtype=torch.float32)
        if old:
            recurrent[:old, :old].copy_(self.recurrent_weight)
            recurrent[old:, :old] = torch.from_numpy(old_to_new)
        recurrent[old:, old:] = torch.diag(torch.from_numpy(self_weights))
        # recurrent[:old, old:] remains zero: candidates cannot alter old features.
        proposed = RandomRSNNState(
            config=self.config,
            input_weight=torch.cat(
                [self.input_weight, torch.from_numpy(win_new)], dim=0
            ),
            recurrent_weight=recurrent,
            bias=torch.cat(
                [
                    self.bias,
                    torch.from_numpy(
                        rng.uniform(-bias_scale, bias_scale, size=pool_size).astype(
                            np.float32
                        )
                    ),
                ]
            ),
        )
        return proposed, scales

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
        )

    def checkpoint(self) -> Dict:
        return {
            "config": asdict(self.config),
            "input_weight": self.input_weight,
            "recurrent_weight": self.recurrent_weight,
            "bias": self.bias,
        }


def recording_steps(total_steps: int, readout_bins: int) -> np.ndarray:
    if readout_bins <= 0 or readout_bins > total_steps:
        raise ValueError("readout_bins must be in [1, total_steps]")
    steps = np.ceil(
        np.arange(1, readout_bins + 1, dtype=np.float64)
        * total_steps
        / readout_bins
    ).astype(np.int64) - 1
    steps[-1] = total_steps - 1
    return steps


@torch.inference_mode()
def extract_readout_traces(
    loader: DataLoader,
    state: RandomRSNNState,
    device: torch.device,
    readout_bins: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return traces [sample, readout_time, hidden], labels, and firing rates."""
    hidden = state.hidden_size
    win = state.input_weight.to(device)
    wrec = state.recurrent_weight.to(device)
    bias = state.bias.to(device)
    syn_decay = math.exp(-state.config.dt / state.config.tau_syn)
    mem_decay = math.exp(-state.config.dt / state.config.tau_mem)
    mem_input_scale = 1.0 - mem_decay
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
            readout_syn = inputs.new_zeros(batch_size, hidden)
            readout_mem = inputs.new_zeros(batch_size, hidden)
            batch_spike_total = inputs.new_zeros(hidden)
            for step in range(total_steps):
                current = inputs[:, step] @ win.t() + spike @ wrec.t() + bias
                syn = syn_decay * syn + current
                # Cramer et al. Eq. (6): u[n+1] = lambda*u[n]*(1-S[n])
                # + (1-lambda)*I[n].  The previous implementation omitted
                # (1-lambda), which severely over-drove neurons at small dt.
                mem = mem_decay * mem + mem_input_scale * syn
                spike = (mem > state.config.threshold).to(inputs.dtype)
                mem = mem * (1.0 - spike)

                # Per-neuron basis whose linear combination is the output trace.
                readout_syn = syn_decay * readout_syn + spike
                readout_mem = (
                    mem_decay * readout_mem + mem_input_scale * readout_syn
                )
                if step in point_to_bin:
                    traces[:, point_to_bin[step], :] = readout_mem
                batch_spike_total += spike.sum(dim=0)
            spike_total += batch_spike_total.double().cpu()
            example_steps += batch_size * total_steps
        all_features.append(traces.cpu())
        all_labels.append(labels.clone())

    rates = spike_total / float(max(1, example_steps))
    return torch.cat(all_features), torch.cat(all_labels), rates


def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    return torch.nn.functional.one_hot(labels.long(), num_classes).double()


def temporal_design(
    traces: Tensor,
    labels: Tensor,
    num_classes: int,
    warmup_bins: int,
) -> Tuple[Tensor, Tensor]:
    if not 0 <= warmup_bins < traces.shape[1]:
        raise ValueError("warmup_bins must be smaller than readout_bins")
    selected = traces[:, warmup_bins:, :].double()
    rows = selected.shape[0] * selected.shape[1]
    features = selected.reshape(rows, selected.shape[2])
    design = torch.cat([torch.ones(rows, 1, dtype=torch.float64), features], dim=1)
    targets = one_hot(labels, num_classes)[:, None, :].expand(
        -1, selected.shape[1], -1
    )
    return design, targets.reshape(rows, num_classes)


def solve_readout(design: Tensor, targets: Tensor, ridge: float) -> Tensor:
    gram = design.t() @ design
    penalty = torch.eye(gram.shape[0], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    rhs = design.t() @ targets
    try:
        return torch.linalg.solve(gram + penalty, rhs)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(gram + penalty, rhs).solution


def max_over_time_scores(traces: Tensor, beta: Tensor) -> Tensor:
    bias = beta[0]
    weights = beta[1:]
    temporal_scores = torch.einsum(
        "nth,hc->ntc", traces.double(), weights
    ) + bias
    return temporal_scores.max(dim=1).values


def classification_metrics(scores: Tensor, labels: Tensor) -> Dict[str, float]:
    probabilities = torch.softmax(scores, dim=1).clamp_min(1e-12)
    loss = -torch.log(probabilities[torch.arange(labels.numel()), labels]).mean()
    accuracy = (scores.argmax(dim=1) == labels).double().mean()
    return {"loss": float(loss), "accuracy": float(accuracy)}


def residualize(design: Tensor, candidates: Tensor, ridge: float) -> Tensor:
    gram = design.t() @ design
    penalty = torch.eye(gram.shape[0], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    rhs = design.t() @ candidates
    coefficients = torch.linalg.solve(gram + penalty, rhs)
    return candidates - design @ coefficients


def greedy_residual_screen(
    base_design: Tensor,
    candidate_columns: Tensor,
    targets: Tensor,
    candidate_rates: Tensor,
    growth_step: int,
    r_values: Sequence[float],
    ridge: float,
    min_rate: float,
    max_rate: float,
    min_gain: float,
) -> Tuple[List[int], List[Dict], Tensor, Tensor]:
    """Greedily select candidates using residualized projection gain."""
    design = base_design
    remaining = list(range(candidate_columns.shape[1]))
    selected: List[int] = []
    records: List[Dict] = []

    for slot in range(min(int(growth_step), candidate_columns.shape[1])):
        beta = solve_readout(design, targets, ridge)
        residual = targets - design @ beta
        residual_energy = float(torch.sum(residual * residual))
        if residual_energy <= 1e-14 or not remaining:
            break
        raw = candidate_columns[:, remaining]
        projected = residualize(design, raw, ridge)
        denominator = torch.sum(projected * projected, dim=0)
        numerator = torch.sum((projected.t() @ residual).pow(2), dim=1)
        gain = numerator / denominator.clamp_min(1e-14)
        rho = gain / residual_energy

        valid_rate = (
            (candidate_rates[remaining] >= float(min_rate))
            & (candidate_rates[remaining] <= float(max_rate))
        )
        chosen_local = None
        chosen_r = None
        for r_value in r_values:
            passing = (
                valid_rate
                & (gain >= float(min_gain))
                & (rho >= 1.0 - float(r_value))
                & (denominator > 1e-14)
            )
            if bool(torch.any(passing)):
                masked = torch.where(
                    passing, rho, torch.full_like(rho, -torch.inf)
                )
                chosen_local = int(torch.argmax(masked))
                chosen_r = float(r_value)
                break
        if chosen_local is None:
            break

        chosen = remaining[chosen_local]
        selected.append(chosen)
        records.append(
            {
                "slot": slot,
                "candidate": chosen,
                "r": chosen_r,
                "gain": float(gain[chosen_local]),
                "rho": float(rho[chosen_local]),
                "rate": float(candidate_rates[chosen]),
            }
        )
        design = torch.cat([design, candidate_columns[:, chosen : chosen + 1]], dim=1)
        remaining.remove(chosen)

    beta = solve_readout(design, targets, ridge)
    return selected, records, design, beta


def selected_traces(
    pool_traces: Tensor, old_hidden: int, selected_relative: Sequence[int]
) -> Tensor:
    indices = list(range(old_hidden)) + [
        old_hidden + int(index) for index in selected_relative
    ]
    return pool_traces[:, :, indices]


def build_loaders(args):
    train_dataset = SHDEvents(args.data_dir / "shd_train.h5")
    test_dataset = SHDEvents(args.data_dir / "shd_test.h5")
    train_indices, val_indices = stratified_split(
        train_dataset.labels, args.val_ratio, args.seed
    )
    collate = SHDBinner(args.dt, args.max_time)
    common = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices.tolist()), **common
    )
    val_loader = DataLoader(Subset(train_dataset, val_indices.tolist()), **common)
    test_loader = DataLoader(test_dataset, **common)
    return (
        train_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_indices,
        val_indices,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure random-screening BDNN-RSNN on SHD; no training and no ZO."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "SHD",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/shd_bdnn_rsnn_pure")
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--max-time", type=float, default=1.4)
    parser.add_argument("--readout-bins", type=int, default=32)
    parser.add_argument("--readout-warmup-bins", type=int, default=4)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--growth-step", type=int, default=8)
    parser.add_argument("--min-selected", type=int, default=1)
    parser.add_argument("--max-hidden", type=int, default=256)
    parser.add_argument("--max-growth-rounds", type=int, default=100)
    parser.add_argument("--growth-failure-patience", type=int, default=10)
    parser.add_argument("--growth-min-delta", type=float, default=0.001)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--r-values",
        type=float,
        nargs="+",
        default=[0.9, 0.99, 0.999, 0.9999],
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-scale", type=float, default=0.5)
    parser.add_argument("--self-recurrent-scale", type=float, default=0.5)
    parser.add_argument("--bias-scale", type=float, default=0.1)
    parser.add_argument("--candidate-min-rate", type=float, default=0.0005)
    parser.add_argument("--candidate-max-rate", type=float, default=0.30)
    parser.add_argument("--min-gain", type=float, default=1e-10)
    parser.add_argument("--tau-syn", type=float, default=0.010)
    parser.add_argument("--tau-mem", type=float, default=0.020)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_pool_size < args.growth_step:
        raise ValueError("candidate_pool_size must be >= growth_step")
    if args.min_selected < 1 or args.min_selected > args.growth_step:
        raise ValueError("min_selected must be in [1, growth_step]")
    total_steps = int(math.ceil(args.max_time / args.dt))
    recording_steps(total_steps, args.readout_bins)
    if not 0 <= args.readout_warmup_bins < args.readout_bins:
        raise ValueError("invalid readout_warmup_bins")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        train_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_indices,
        val_indices,
    ) = build_loaders(args)

    config = DynamicsConfig(
        dt=args.dt,
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
        threshold=args.threshold,
    )
    state = RandomRSNNState.empty(config)
    train_traces, train_labels, _ = extract_readout_traces(
        train_loader, state, device, args.readout_bins
    )
    val_traces, val_labels, _ = extract_readout_traces(
        val_loader, state, device, args.readout_bins
    )
    base_design, temporal_targets = temporal_design(
        train_traces,
        train_labels,
        num_classes=20,
        warmup_bins=args.readout_warmup_bins,
    )
    beta = solve_readout(base_design, temporal_targets, args.ridge)
    current_val = classification_metrics(
        max_over_time_scores(val_traces, beta), val_labels
    )
    print(
        "device={} train={} val={} test={} time_steps={} readout_bins={} "
        "initial_hidden=0 val_acc={:.4f}".format(
            device,
            len(train_indices),
            len(val_indices),
            len(test_dataset),
            total_steps,
            args.readout_bins,
            current_val["accuracy"],
        ),
        flush=True,
    )

    history: List[Dict] = []
    failures = 0
    for round_index in range(1, args.max_growth_rounds + 1):
        if state.hidden_size >= args.max_hidden:
            break
        pool_size = min(
            args.candidate_pool_size,
            args.max_hidden - state.hidden_size,
        )
        keep_limit = min(args.growth_step, pool_size)
        rng = np.random.RandomState(args.seed + round_index * 1009)
        pool_state, scales = state.propose_pool(
            pool_size=pool_size,
            lambdas=args.lambdas,
            rng=rng,
            input_scale=args.input_scale,
            recurrent_scale=args.recurrent_scale,
            self_recurrent_scale=args.self_recurrent_scale,
            bias_scale=args.bias_scale,
        )
        old_hidden = state.hidden_size
        pool_train_traces, labels_check, pool_rates = extract_readout_traces(
            train_loader, pool_state, device, args.readout_bins
        )
        if not torch.equal(labels_check, train_labels):
            raise RuntimeError("training sample order changed during feature extraction")
        pool_design, _ = temporal_design(
            pool_train_traces,
            train_labels,
            num_classes=20,
            warmup_bins=args.readout_warmup_bins,
        )
        candidate_columns = pool_design[:, 1 + old_hidden :]
        selected, screen_records, selected_design, selected_beta = greedy_residual_screen(
            base_design=base_design,
            candidate_columns=candidate_columns,
            targets=temporal_targets,
            candidate_rates=pool_rates[old_hidden:],
            growth_step=keep_limit,
            r_values=args.r_values,
            ridge=args.ridge,
            min_rate=args.candidate_min_rate,
            max_rate=args.candidate_max_rate,
            min_gain=args.min_gain,
        )

        accepted = False
        proposal_val = None
        proposal_state = None
        proposal_val_traces = None
        if len(selected) >= args.min_selected:
            proposal_state = pool_state.select_from_pool(old_hidden, selected)
            pool_val_traces, val_labels_check, _ = extract_readout_traces(
                val_loader, pool_state, device, args.readout_bins
            )
            if not torch.equal(val_labels_check, val_labels):
                raise RuntimeError("validation sample order changed")
            proposal_val_traces = selected_traces(
                pool_val_traces, old_hidden, selected
            )
            proposal_val = classification_metrics(
                max_over_time_scores(proposal_val_traces, selected_beta), val_labels
            )
            accepted = proposal_val["accuracy"] > (
                current_val["accuracy"] + args.growth_min_delta
            )

        event = {
            "round": round_index,
            "old_hidden": old_hidden,
            "pool_size": pool_size,
            "selected": selected,
            "selected_scales": [float(scales[index]) for index in selected],
            "screen": screen_records,
            "base_validation": current_val,
            "proposal_validation": proposal_val,
            "accepted": accepted,
        }
        history.append(event)
        if accepted:
            state = proposal_state
            train_traces = selected_traces(
                pool_train_traces, old_hidden, selected
            )
            val_traces = proposal_val_traces
            base_design = selected_design
            beta = selected_beta
            current_val = proposal_val
            failures = 0
            print(
                "[commit] round={} pool={} selected={} hidden={} val_acc={:.4f}".format(
                    round_index,
                    pool_size,
                    len(selected),
                    state.hidden_size,
                    current_val["accuracy"],
                ),
                flush=True,
            )
        else:
            failures += 1
            proposal_text = (
                "none" if proposal_val is None else "{:.4f}".format(
                    proposal_val["accuracy"]
                )
            )
            print(
                "[rollback] round={} pool={} passed={} proposal_val={} "
                "base_val={:.4f} failures={}/{}".format(
                    round_index,
                    pool_size,
                    len(selected),
                    proposal_text,
                    current_val["accuracy"],
                    failures,
                    args.growth_failure_patience,
                ),
                flush=True,
            )
            if failures >= args.growth_failure_patience:
                break

    # The official test set is used exactly once, after structure selection.
    test_traces, test_labels, test_rates = extract_readout_traces(
        test_loader, state, device, args.readout_bins
    )
    test_metrics = classification_metrics(
        max_over_time_scores(test_traces, beta), test_labels
    )
    checkpoint_path = args.output_dir / "final_pure_bdnn_rsnn.pt"
    torch.save(
        {
            "state": state.checkpoint(),
            "beta": beta,
            "args": vars(args),
            "label_keys": train_dataset.keys,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "validation": current_val,
            "test": test_metrics,
            "test_rates": test_rates,
            "history": history,
        },
        checkpoint_path,
    )
    summary = {
        "hidden_size": state.hidden_size,
        "validation": current_val,
        "test": test_metrics,
        "accepted_rounds": sum(int(item["accepted"]) for item in history),
        "attempted_rounds": len(history),
        "checkpoint": str(checkpoint_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "growth_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
