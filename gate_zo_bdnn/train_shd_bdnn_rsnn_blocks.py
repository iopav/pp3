"""Strict pure-BDNN block-growing RSNN for SHD.

This program deliberately contains no gradient, optimizer, epoch loop, BPTT,
or zeroth-order update.  A growth round generates several independent random
recurrent blocks, evaluates every block with a closed-form temporal readout,
and commits the block that performs best under the same max-over-time rule used
for inference.

Committed neurons never receive feedback from younger blocks.  This preserves
the old temporal features exactly across growth rounds.  Neurons inside a new
block are recurrently connected, so the unit of selection is a temporal
feature-generating subnetwork rather than an isolated random neuron.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from . import train_shd_bdnn_rsnn_pure as core
except ImportError:
    import train_shd_bdnn_rsnn_pure as core


class DeterministicJitterAugmentedSubset(Dataset):
    """Original samples plus reproducible channel-jittered training copies.

    A fixed perturbation is used for every source-sample/copy pair.  This is
    important for constructive training because committed traces and candidate
    traces are extracted in separate loader passes and must remain aligned.
    """

    def __init__(
        self,
        dataset,
        indices,
        jitter_std: float,
        jitter_copies: int,
        seed: int,
        source_channels: int = 700,
        time_scale_min: float = 1.0,
        time_scale_max: float = 1.0,
    ):
        if jitter_std < 0.0 or jitter_copies <= 0:
            raise ValueError("jitter_std must be nonnegative and copies positive")
        if time_scale_min <= 0.0 or time_scale_max < time_scale_min:
            raise ValueError("invalid time-scale augmentation range")
        if jitter_std == 0.0 and time_scale_min == 1.0 and time_scale_max == 1.0:
            raise ValueError("at least one augmentation must be active")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.jitter_std = float(jitter_std)
        self.jitter_copies = int(jitter_copies)
        self.seed = int(seed)
        self.source_channels = int(source_channels)
        self.time_scale_min = float(time_scale_min)
        self.time_scale_max = float(time_scale_max)
        base_labels = np.asarray(dataset.labels)[self.indices]
        self.labels = np.tile(base_labels, self.jitter_copies + 1)

    def __len__(self):
        return int(len(self.indices) * (self.jitter_copies + 1))

    def __getitem__(self, index):
        base_count = len(self.indices)
        copy_index, local_index = divmod(int(index), base_count)
        source_index = int(self.indices[local_index])
        times, units, label = self.dataset[source_index]
        if copy_index:
            jitter_seed = (
                self.seed
                + source_index * 1_000_003
                + copy_index * 97_409
            ) % (2**32 - 1)
            rng = np.random.RandomState(jitter_seed)
            if self.jitter_std > 0.0:
                offsets = np.rint(
                    rng.normal(0.0, self.jitter_std, size=units.shape)
                ).astype(np.int64)
                units = np.clip(
                    units.astype(np.int64, copy=False) + offsets,
                    0,
                    self.source_channels - 1,
                )
            if self.time_scale_min != 1.0 or self.time_scale_max != 1.0:
                scale = rng.uniform(self.time_scale_min, self.time_scale_max)
                times = np.asarray(times, dtype=np.float64) * float(scale)
        return times, units, label


def speaker_disjoint_split(
    speakers: np.ndarray, validation_speakers: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Hold out complete speakers so validation measures speaker transfer."""
    speakers = np.asarray(speakers).astype(str)
    unique = np.unique(speakers)
    if validation_speakers < 1 or validation_speakers >= unique.size:
        raise ValueError("validation_speakers must be in [1, number_of_speakers-1]")
    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    held_out = sorted(shuffled[:validation_speakers].tolist())
    val_mask = np.isin(speakers, held_out)
    train_indices = np.flatnonzero(~val_mask)
    val_indices = np.flatnonzero(val_mask)
    if train_indices.size == 0 or val_indices.size == 0:
        raise RuntimeError("speaker split produced an empty partition")
    return train_indices, val_indices, held_out


def build_speaker_loaders(args):
    train_dataset = core.SHDEvents(args.data_dir / "shd_train.h5")
    test_dataset = core.SHDEvents(args.data_dir / "shd_test.h5")
    if train_dataset.speakers is None:
        raise RuntimeError("SHD file has no extra/speaker metadata")
    train_indices, val_indices, held_out = speaker_disjoint_split(
        train_dataset.speakers, args.validation_speakers, args.seed
    )
    import os
    if os.environ.get('PP3_CONFIG'):
        import json
        from pp3.splits import outer_split
        shared_config = json.loads(Path(os.environ['PP3_CONFIG']).read_text(encoding='utf-8'))
        train_indices, val_indices = outer_split(train_dataset.labels, shared_config, train_dataset.speakers)
        held_out = sorted(np.unique(train_dataset.speakers[val_indices]).tolist())
    smoke_limit = getattr(args, 'pp3_smoke_samples', None)
    if smoke_limit:
        train_indices, val_indices = train_indices[:smoke_limit], val_indices[:smoke_limit]
        test_dataset = Subset(test_dataset, list(range(min(smoke_limit, len(test_dataset)))))
    channel_group_size = int(getattr(args, "input_channel_group_size", 1))
    jitter_std = float(getattr(args, "train_channel_jitter_std", 0.0))
    jitter_copies = int(getattr(args, "train_jitter_copies", 0))
    time_scale_min = float(getattr(args, "train_time_scale_min", 1.0))
    time_scale_max = float(getattr(args, "train_time_scale_max", 1.0))
    collate = core.SHDBinner(
        args.dt,
        args.max_time,
        channels=700,
        channel_group_size=channel_group_size,
    )
    common = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    augmentation_active = (
        jitter_std > 0.0 or time_scale_min != 1.0 or time_scale_max != 1.0
    )
    if augmentation_active and jitter_copies > 0:
        training_subset = DeterministicJitterAugmentedSubset(
            train_dataset,
            train_indices,
            jitter_std,
            jitter_copies,
            args.seed + 65_537,
            time_scale_min=time_scale_min,
            time_scale_max=time_scale_max,
        )
    else:
        training_subset = Subset(train_dataset, train_indices.tolist())
    train_loader = DataLoader(training_subset, **common)
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
        held_out,
    )


def block_slices(old_hidden: int, block_size: int, block_count: int) -> List[slice]:
    return [
        slice(old_hidden + index * block_size, old_hidden + (index + 1) * block_size)
        for index in range(block_count)
    ]


def normalized_sparse_recurrent(
    size: int,
    density: float,
    spectral_radius: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Create one random recurrent block with a controlled spectral radius."""
    # A sparse directed matrix can contain nonzero edges yet be acyclic.  Such
    # a matrix is nilpotent and has spectral radius zero, so merely checking
    # ``np.any(weights)`` is insufficient.  Resample a bounded number of times
    # to obtain a genuinely recurrent graph.  The diagonal fallback guarantees
    # termination for very small blocks/densities without aborting a long run.
    radius = 0.0
    for _ in range(16):
        weights = rng.standard_normal((size, size)).astype(np.float32)
        mask = rng.random_sample((size, size)) < float(density)
        weights *= mask.astype(np.float32)
        if np.any(weights):
            radius = float(
                np.max(np.abs(np.linalg.eigvals(weights.astype(np.float64))))
            )
            if math.isfinite(radius) and radius > 1e-12:
                break
    else:
        weights = np.zeros((size, size), dtype=np.float32)
        weights[0, 0] = 1.0
        radius = 1.0
    weights *= float(spectral_radius) / radius
    return weights


def propose_block_pool(
    state: core.RandomRSNNState,
    block_count: int,
    block_size: int,
    input_scales: Sequence[float],
    spectral_radii: Sequence[float],
    recurrent_density: float,
    old_to_new_scale: float,
    bias_scale: float,
    rng: np.random.RandomState,
) -> Tuple[core.RandomRSNNState, List[Dict]]:
    """Append independent recurrent candidate blocks to a committed state."""
    if block_count <= 0 or block_size <= 0:
        raise ValueError("block_count and block_size must be positive")
    if not input_scales or not spectral_radii:
        raise ValueError("candidate scale lists must be non-empty")
    if not 0.0 < recurrent_density <= 1.0:
        raise ValueError("recurrent_density must be in (0, 1]")

    old = state.hidden_size
    added = block_count * block_size
    total = old + added
    win_new = np.empty((added, state.config.input_size), dtype=np.float32)
    old_to_new = np.zeros((added, old), dtype=np.float32)
    recurrent_new = np.zeros((added, added), dtype=np.float32)
    bias_new = rng.uniform(-bias_scale, bias_scale, size=added).astype(np.float32)
    metadata: List[Dict] = []

    # Draw every candidate independently.  Resizing the lists made a one-block
    # pool always receive the first (and formerly smallest) scale.
    scale_choices = rng.choice(
        np.asarray(input_scales, dtype=np.float64), size=block_count, replace=True
    )
    radius_choices = rng.choice(
        np.asarray(spectral_radii, dtype=np.float64), size=block_count, replace=True
    )

    for block_index in range(block_count):
        start = block_index * block_size
        stop = start + block_size
        input_scale = float(scale_choices[block_index])
        radius = float(radius_choices[block_index])
        input_limit = input_scale / math.sqrt(state.config.input_size)
        win_new[start:stop] = rng.uniform(
            -input_limit,
            input_limit,
            size=(block_size, state.config.input_size),
        )
        if old:
            recurrent_limit = float(old_to_new_scale) / math.sqrt(old)
            old_to_new[start:stop] = rng.uniform(
                -recurrent_limit,
                recurrent_limit,
                size=(block_size, old),
            )
        recurrent_new[start:stop, start:stop] = normalized_sparse_recurrent(
            block_size,
            density=recurrent_density,
            spectral_radius=radius,
            rng=rng,
        )
        metadata.append(
            {
                "block": block_index,
                "input_scale": input_scale,
                "spectral_radius": radius,
            }
        )

    recurrent = torch.zeros(total, total, dtype=torch.float32)
    if old:
        recurrent[:old, :old].copy_(state.recurrent_weight)
        recurrent[old:, :old] = torch.from_numpy(old_to_new)
    recurrent[old:, old:] = torch.from_numpy(recurrent_new)
    # recurrent[:old, old:] is intentionally zero: new blocks cannot perturb old ones.
    proposed = core.RandomRSNNState(
        config=state.config,
        input_weight=torch.cat([state.input_weight, torch.from_numpy(win_new)]),
        recurrent_weight=recurrent,
        bias=torch.cat([state.bias, torch.from_numpy(bias_new)]),
    )
    return proposed, metadata


def select_block_state(
    pool_state: core.RandomRSNNState,
    old_hidden: int,
    chosen: slice,
) -> core.RandomRSNNState:
    relative = np.arange(chosen.start - old_hidden, chosen.stop - old_hidden)
    return pool_state.select_from_pool(old_hidden, relative)


def select_block_traces(pool_traces: Tensor, old_hidden: int, chosen: slice) -> Tensor:
    indices = list(range(old_hidden)) + list(range(chosen.start, chosen.stop))
    return pool_traces[:, :, indices]


def standardize_temporal_features(
    train_traces: Tensor, val_traces: Tensor, warmup_bins: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    fit = train_traces[:, warmup_bins:, :].double()
    if fit.shape[2] == 0:
        mean = torch.empty(0, dtype=torch.float64)
        scale = torch.empty(0, dtype=torch.float64)
    else:
        mean = fit.mean(dim=(0, 1))
        scale = fit.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    train_normalized = (train_traces.double() - mean) / scale
    val_normalized = (val_traces.double() - mean) / scale
    return train_normalized, val_normalized, mean, scale


def solve_mean_normalized_ridge(design: Tensor, targets: Tensor, ridge: float) -> Tensor:
    """Solve mean-squared ridge so ridge is independent of row count."""
    rows = max(1, design.shape[0])
    gram = (design.t() @ design) / float(rows)
    rhs = (design.t() @ targets) / float(rows)
    penalty = torch.eye(gram.shape[0], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    try:
        return torch.linalg.solve(gram + penalty, rhs)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(gram + penalty, rhs).solution


def fit_and_measure(
    train_traces: Tensor,
    train_labels: Tensor,
    val_traces: Tensor,
    val_labels: Tensor,
    warmup_bins: int,
    ridge: float,
) -> Tuple[Tensor, Dict[str, float], Dict[str, float], Tensor, Tensor]:
    normalized_train, normalized_val, feature_mean, feature_scale = (
        standardize_temporal_features(train_traces, val_traces, warmup_bins)
    )
    design, targets = core.temporal_design(
        normalized_train, train_labels, num_classes=20, warmup_bins=warmup_bins
    )
    beta = solve_mean_normalized_ridge(design, targets, ridge)
    train_metrics = core.classification_metrics(
        core.max_over_time_scores(normalized_train[:, warmup_bins:], beta),
        train_labels,
    )
    val_metrics = core.classification_metrics(
        core.max_over_time_scores(normalized_val[:, warmup_bins:], beta), val_labels
    )
    return beta, train_metrics, val_metrics, feature_mean, feature_scale


def candidate_is_better(
    candidate: Dict[str, float], current: Dict[str, float], min_accuracy_gain: float
) -> bool:
    accuracy_gain = candidate["accuracy"] - current["accuracy"]
    if accuracy_gain > float(min_accuracy_gain):
        return True
    # Accuracy is discrete.  At an exact tie, permit a strictly better aligned
    # max-over-time cross-entropy to guide growth without inventing an epsilon.
    return abs(accuracy_gain) <= 1e-12 and candidate["loss"] < current["loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/shd_bdnn_rsnn_blocks")
    )
    # Strict paper-scale temporal discretization; no coarse fallback default.
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--readout-bins", type=int, default=100)
    parser.add_argument("--readout-warmup-bins", type=int, default=5)
    parser.add_argument("--candidate-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-hidden", type=int, default=1024)
    parser.add_argument("--max-growth-rounds", type=int, default=64)
    parser.add_argument("--growth-failure-patience", type=int, default=20)
    parser.add_argument("--growth-min-delta", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--input-scales", type=float, nargs="+", default=[2.0, 3.0, 4.0, 6.0]
    )
    parser.add_argument(
        "--spectral-radii", type=float, nargs="+", default=[0.7, 0.9, 1.1, 1.3]
    )
    parser.add_argument("--recurrent-density", type=float, default=0.1)
    parser.add_argument("--old-to-new-scale", type=float, default=0.5)
    parser.add_argument("--bias-scale", type=float, default=0.0)
    parser.add_argument("--candidate-min-rate", type=float, default=0.0005)
    parser.add_argument("--candidate-max-rate", type=float, default=0.30)
    parser.add_argument("--min-active-fraction", type=float, default=0.25)
    # The paper's best generalization setting scales 10/20 ms by four.
    parser.add_argument("--tau-syn", type=float, default=0.040)
    parser.add_argument("--tau-mem", type=float, default=0.080)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--validation-speakers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from output-dir/latest_growth.pt if it exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_hidden < args.block_size:
        raise ValueError("max_hidden must be at least one block")
    if args.max_hidden % args.block_size:
        raise ValueError("max_hidden must be divisible by block_size")
    if not 0.0 <= args.min_active_fraction <= 1.0:
        raise ValueError("min_active_fraction must be in [0, 1]")
    core.recording_steps(int(math.ceil(args.max_time / args.dt)), args.readout_bins)
    if not 0 <= args.readout_warmup_bins < args.readout_bins:
        raise ValueError("invalid readout warmup")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

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
        held_out_speakers,
    ) = build_speaker_loaders(args)

    config = core.DynamicsConfig(
        dt=args.dt,
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
        threshold=args.threshold,
    )
    latest_path = args.output_dir / "latest_growth.pt"
    history: List[Dict] = []
    failures = 0
    start_round = 1
    if args.resume and latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu")
        saved = latest["state"]
        state = core.RandomRSNNState(
            config=config,
            input_weight=saved["input_weight"].float(),
            recurrent_weight=saved["recurrent_weight"].float(),
            bias=saved["bias"].float(),
        )
        history = latest.get("history", [])
        failures = int(latest.get("failures", 0))
        start_round = int(latest.get("next_round", len(history) + 1))
        print(
            "[resume] path={} hidden={} next_round={} failures={}".format(
                latest_path, state.hidden_size, start_round, failures
            ),
            flush=True,
        )
    else:
        state = core.RandomRSNNState.empty(config)
    train_traces, train_labels, _ = core.extract_readout_traces(
        train_loader, state, device, args.readout_bins
    )
    val_traces, val_labels, _ = core.extract_readout_traces(
        val_loader, state, device, args.readout_bins
    )
    (
        beta,
        current_train,
        current_val,
        feature_mean,
        feature_scale,
    ) = fit_and_measure(
        train_traces,
        train_labels,
        val_traces,
        val_labels,
        args.readout_warmup_bins,
        args.ridge,
    )
    print(
        "device={} train={} val={} test={} time_steps={} readout_bins={} "
        "block_size={} candidate_blocks={} initial_hidden={} val_acc={:.4f} "
        "val_loss={:.4f} held_out_speakers={}".format(
            device,
            len(train_indices),
            len(val_indices),
            len(test_dataset),
            int(math.ceil(args.max_time / args.dt)),
            args.readout_bins,
            args.block_size,
            args.candidate_blocks,
            state.hidden_size,
            current_val["accuracy"],
            current_val["loss"],
            held_out_speakers,
        ),
        flush=True,
    )

    for round_index in range(start_round, args.max_growth_rounds + 1):
        remaining = args.max_hidden - state.hidden_size
        if remaining < args.block_size:
            break
        # max_hidden constrains committed width, not the temporary search pool.
        candidate_blocks = args.candidate_blocks
        old_hidden = state.hidden_size
        rng = np.random.RandomState(args.seed + round_index * 1009)
        pool_state, metadata = propose_block_pool(
            state,
            block_count=candidate_blocks,
            block_size=args.block_size,
            input_scales=args.input_scales,
            spectral_radii=args.spectral_radii,
            recurrent_density=args.recurrent_density,
            old_to_new_scale=args.old_to_new_scale,
            bias_scale=args.bias_scale,
            rng=rng,
        )
        pool_train, train_check, pool_rates = core.extract_readout_traces(
            train_loader, pool_state, device, args.readout_bins
        )
        pool_val, val_check, _ = core.extract_readout_traces(
            val_loader, pool_state, device, args.readout_bins
        )
        if not torch.equal(train_check, train_labels) or not torch.equal(
            val_check, val_labels
        ):
            raise RuntimeError("sample order changed during candidate extraction")

        records: List[Dict] = []
        best = None
        for block_index, candidate_slice in enumerate(
            block_slices(old_hidden, args.block_size, candidate_blocks)
        ):
            rates = pool_rates[candidate_slice]
            active_fraction = float(
                ((rates >= args.candidate_min_rate) & (rates <= args.candidate_max_rate))
                .double()
                .mean()
            )
            candidate_train = select_block_traces(
                pool_train, old_hidden, candidate_slice
            )
            candidate_val = select_block_traces(pool_val, old_hidden, candidate_slice)
            (
                candidate_beta,
                train_metrics,
                val_metrics,
                candidate_mean,
                candidate_scale,
            ) = fit_and_measure(
                candidate_train,
                train_labels,
                candidate_val,
                val_labels,
                args.readout_warmup_bins,
                args.ridge,
            )
            valid = active_fraction >= args.min_active_fraction
            record = {
                **metadata[block_index],
                "active_fraction": active_fraction,
                "mean_rate": float(rates.mean()),
                "train": train_metrics,
                "validation": val_metrics,
                "valid": valid,
            }
            records.append(record)
            rank = (val_metrics["accuracy"], -val_metrics["loss"])
            if valid and (best is None or rank > best["rank"]):
                best = {
                    "rank": rank,
                    "block": block_index,
                    "slice": candidate_slice,
                    "train_traces": candidate_train,
                    "val_traces": candidate_val,
                    "beta": candidate_beta,
                    "feature_mean": candidate_mean,
                    "feature_scale": candidate_scale,
                    "train": train_metrics,
                    "validation": val_metrics,
                }

        accepted = best is not None and candidate_is_better(
            best["validation"], current_val, args.growth_min_delta
        )
        event = {
            "round": round_index,
            "old_hidden": old_hidden,
            "candidates": records,
            "chosen_block": None if best is None else best["block"],
            "base_validation": current_val,
            "accepted": bool(accepted),
        }
        history.append(event)
        rate_summary = "; ".join(
            "b{}:in={:.1f},rho={:.1f},active={:.2f},mean={:.5f}".format(
                item["block"],
                item["input_scale"],
                item["spectral_radius"],
                item["active_fraction"],
                item["mean_rate"],
            )
            for item in records
        )
        if accepted:
            chosen = best["slice"]
            state = select_block_state(pool_state, old_hidden, chosen)
            train_traces = best["train_traces"]
            val_traces = best["val_traces"]
            beta = best["beta"]
            feature_mean = best["feature_mean"]
            feature_scale = best["feature_scale"]
            current_train = best["train"]
            current_val = best["validation"]
            failures = 0
            print(
                "[commit] round={} candidates={} chosen={} hidden={} "
                "train_acc={:.4f} val_acc={:.4f} val_loss={:.4f} rates=[{}]".format(
                    round_index,
                    candidate_blocks,
                    best["block"],
                    state.hidden_size,
                    current_train["accuracy"],
                    current_val["accuracy"],
                    current_val["loss"],
                    rate_summary,
                ),
                flush=True,
            )
        else:
            failures += 1
            proposal = None if best is None else best["validation"]
            print(
                "[rollback] round={} candidates={} best_val={} base_val={:.4f} "
                "failures={}/{} rates=[{}]".format(
                    round_index,
                    candidate_blocks,
                    "none" if proposal is None else "{:.4f}".format(proposal["accuracy"]),
                    current_val["accuracy"],
                    failures,
                    args.growth_failure_patience,
                    rate_summary,
                ),
                flush=True,
            )
            if failures >= args.growth_failure_patience:
                torch.save(
                    {
                        "state": state.checkpoint(),
                        "history": history,
                        "failures": failures,
                        "next_round": round_index + 1,
                    },
                    latest_path,
                )
                break

        torch.save(
            {
                "state": state.checkpoint(),
                "history": history,
                "failures": failures,
                "next_round": round_index + 1,
            },
            latest_path,
        )

    test_traces, test_labels, test_rates = core.extract_readout_traces(
        test_loader, state, device, args.readout_bins
    )
    normalized_test = (test_traces.double() - feature_mean) / feature_scale
    test_metrics = core.classification_metrics(
        core.max_over_time_scores(
            normalized_test[:, args.readout_warmup_bins :], beta
        ),
        test_labels,
    )
    checkpoint_path = args.output_dir / "final_pure_bdnn_block_rsnn.pt"
    torch.save(
        {
            "state": state.checkpoint(),
            "beta": beta,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "args": vars(args),
            "label_keys": train_dataset.keys,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "held_out_speakers": held_out_speakers,
            "training": current_train,
            "validation": current_val,
            "test": test_metrics,
            "test_rates": test_rates,
            "history": history,
        },
        checkpoint_path,
    )
    summary = {
        "hidden_size": state.hidden_size,
        "training": current_train,
        "validation": current_val,
        "test": test_metrics,
        "accepted_rounds": sum(int(item["accepted"]) for item in history),
        "attempted_rounds": len(history),
        "checkpoint": str(checkpoint_path),
        "held_out_speakers": held_out_speakers,
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
