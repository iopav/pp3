"""One-step coordinate zeroth-order structural selection for BDNN.

This module does not perturb or optimize continuous weights.  After BDNN has
proposed a block of neurons, it evaluates the full block and one hard ablation
per new neuron.  The ablation losses estimate each neuron's coordinate-wise
zero-order importance.  Exact prefix evaluations then choose the block size
under an L0-penalized calibration objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def clone_value(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return np.array(value, copy=True)


def restore_like(array: np.ndarray, reference):
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    return np.array(array, copy=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    labels = np.argmax(targets, axis=1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True) + 1e-12)
    return float(-np.mean(log_probs[np.arange(labels.size), labels]))


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / (values.sum(axis=1, keepdims=True) + 1e-12)


def _counter(model):
    runtime = getattr(model, '_pp3_runtime', None)
    return runtime.counter if runtime is not None else None


def _matmul(counter, left, right):
    if counter is None:
        return left @ right
    from pp3.numpy_linalg import matmul
    return matmul(counter, left, right)


def _pinv(counter, matrix):
    if counter is None:
        return np.linalg.pinv(matrix)
    from pp3.numpy_linalg import pinv
    return pinv(counter, matrix)


def _solve(counter, matrix, rhs):
    if counter is None:
        return np.linalg.solve(matrix, rhs)
    from pp3.numpy_linalg import solve
    return solve(counter, matrix, rhs)


def fit_readout(features: np.ndarray, targets: np.ndarray, ridge: float, counter=None) -> np.ndarray:
    feature_count = features.shape[1]
    if feature_count == 0:
        return np.zeros((0, targets.shape[1]), dtype=np.float64)
    if ridge <= 0:
        return _matmul(counter, _pinv(counter, features), targets)
    gram = _matmul(counter, features.T, features)
    rhs = _matmul(counter, features.T, targets)
    try:
        return _solve(counter, gram + ridge * np.eye(feature_count), rhs)
    except np.linalg.LinAlgError:
        return _matmul(counter, _pinv(counter, gram + ridge * np.eye(feature_count)), rhs)


def classification_metrics(features: np.ndarray, targets: np.ndarray, beta: np.ndarray, counter=None):
    logits = _matmul(counter, features, beta)
    probs = probabilities(logits)
    return {
        "loss": cross_entropy(logits, targets),
        "mse": float(np.mean((targets - probs) ** 2)),
        "accuracy": float(
            np.mean(np.argmax(logits, axis=1) == np.argmax(targets, axis=1))
        ),
    }


def stratified_split(targets, validation_ratio: float, rng: np.random.Generator):
    """Return disjoint fit/validation indices, preserving every class if possible."""
    labels = np.argmax(np.asarray(targets), axis=1)
    fit, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if indices.size <= 1:
            fit.extend(indices.tolist())
            continue
        count = int(round(indices.size * validation_ratio))
        count = min(max(1, count), indices.size - 1)
        validation.extend(indices[:count].tolist())
        fit.extend(indices[count:].tolist())
    rng.shuffle(fit)
    rng.shuffle(validation)
    if not validation:
        raise ValueError("gate-ZO needs at least two samples in one class")
    return np.asarray(fit, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def balanced_subsample(indices, targets, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if count is None or count <= 0 or count >= indices.size:
        return indices
    labels = np.argmax(np.asarray(targets), axis=1)
    groups = {label: indices[labels[indices] == label].tolist() for label in np.unique(labels[indices])}
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        advanced = False
        for label in sorted(groups):
            if groups[label] and len(chosen) < count:
                chosen.append(groups[label].pop())
                advanced = True
        if not advanced:
            break
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


@dataclass
class GateZOConfig:
    selection_mode: str = "gatezo"
    growth_protocol: str = "legacy_test"
    min_keep: int = 1
    sparsity_lambda: float = 1e-4
    loss_tolerance: float | None = None
    adaptive_tolerance: float = 1e-3
    sample_size: int | None = None
    calibration_ratio: float = 0.5
    ridge: float = 0.0
    validation_ratio: float = 0.2
    growth_min_delta: float = 1e-4 #这俩疑似没用的超参
    failure_patience: int = 10
    seed: int | None = None
    verbose: bool = True

    def validate(self):
        if self.selection_mode not in {"fixed_nozo", "adaptive_nozo", "gatezo"}:
            raise ValueError("selection_mode must be fixed_nozo, adaptive_nozo, or gatezo")
        if self.growth_protocol not in {"legacy_test", "validation_checkpoint"}:
            raise ValueError("growth_protocol must be legacy_test or validation_checkpoint")
        if self.min_keep <= 0:
            raise ValueError("gate_zo_min_keep must be positive")
        if self.sparsity_lambda < 0:
            raise ValueError("gate_zo_sparsity_lambda cannot be negative")
        if self.loss_tolerance is not None and self.loss_tolerance < 0:
            raise ValueError("gate_zo_loss_tolerance cannot be negative")
        if self.adaptive_tolerance < 0:
            raise ValueError("adaptive_tolerance cannot be negative")
        if not 0 < self.calibration_ratio < 1:
            raise ValueError("gate_zo_calibration_ratio must be in (0, 1)")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("gate_zo_validation_ratio must be in (0, 1)")
        if self.failure_patience <= 0:
            raise ValueError("gate_zo_failure_patience must be positive")


class StructuralGateZO:
    """Stateful selector shared by the stage-1 and stage-2 models."""

    def __init__(self, config: GateZOConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.history: List[dict] = []

    def make_outer_split(self, targets):
        return stratified_split(targets, self.config.validation_ratio, self.rng)

    @staticmethod
    def snapshot(model):
        return {
            "W": clone_value(model.W),
            "U": clone_value(model.U),
            "Beta": clone_value(model.Beta) if not isinstance(model.Beta, list) else [],
            "L": int(model.L),
        }

    @staticmethod
    def restore(model, state):
        model.W = clone_value(state["W"])
        model.U = clone_value(state["U"])
        model.Beta = clone_value(state["Beta"]) if not isinstance(state["Beta"], list) else []
        model.L = int(state["L"])

    def _set_mask(self, model, full_w, full_u, old_count, keep):
        keep_global = list(range(old_count)) + [old_count + int(index) for index in keep]
        w = full_w[:, keep_global]
        u = full_u[np.ix_(keep_global, keep_global)]
        model.W = restore_like(w, model.W)
        model.U = restore_like(u, model.U)

    def _evaluate_mask(
        self, model, full_w, full_u, old_count, keep, X, T, fit_indices, cal_indices
    ):
        self._set_mask(model, full_w, full_u, old_count, keep)
        fit_h, _ = model.getH(X[fit_indices])
        cal_h, _ = model.getH(X[cal_indices])
        beta = fit_readout(np.asarray(fit_h), T[fit_indices], self.config.ridge, _counter(model))
        metrics = classification_metrics(np.asarray(cal_h), T[cal_indices], beta, _counter(model))
        return {
            "keep": [int(index) for index in keep],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
        }

    def select(self, model, X, T, old_count: int, proposed_count: int):
        """Select and commit a proposed block under the configured method."""
        if proposed_count <= 0:
            raise ValueError("proposed_count must be positive")
        full_w = np.array(as_numpy(model.W), copy=True)
        full_u = np.array(as_numpy(model.U), copy=True)
        expected = old_count + proposed_count
        if full_w.shape[1] != expected or full_u.shape != (expected, expected):
            raise ValueError("model dimensions do not match the proposed gate-ZO block")

        all_keep = list(range(proposed_count))
        if self.config.selection_mode == "fixed_nozo":
            return {
                "mode": "fixed_nozo",
                "proposed": proposed_count,
                "kept": proposed_count,
                "keep": all_keep,
                "coordinate_queries": 0,
            }

        all_indices = np.arange(X.shape[0], dtype=np.int64)
        sampled = balanced_subsample(
            all_indices, T, self.config.sample_size, self.rng
        )
        local_fit, local_cal = stratified_split(
            T[sampled], self.config.calibration_ratio, self.rng
        )
        fit_indices, cal_indices = sampled[local_fit], sampled[local_cal]

        if self.config.selection_mode == "adaptive_nozo":
            candidates = []
            minimum = min(max(1, self.config.min_keep), proposed_count)
            for count in range(minimum, proposed_count + 1):
                candidates.append(self._evaluate_mask(
                    model, full_w, full_u, old_count, all_keep[:count],
                    X, T, fit_indices, cal_indices,
                ))
            best_loss = min(item["loss"] for item in candidates)
            eligible = [
                item for item in candidates
                if item["loss"] <= best_loss + self.config.adaptive_tolerance
            ]
            chosen = min(eligible, key=lambda item: (len(item["keep"]), item["loss"]))
            self._set_mask(model, full_w, full_u, old_count, chosen["keep"])
            return {
                "mode": "adaptive_nozo",
                "proposed": proposed_count,
                "kept": len(chosen["keep"]),
                "keep": chosen["keep"],
                "coordinate_queries": 0,
                "prefix_evaluations": len(candidates),
                "adaptive_tolerance": self.config.adaptive_tolerance,
                "best_loss": best_loss,
                "chosen_loss": chosen["loss"],
                "chosen_accuracy": chosen["accuracy"],
                "fit_samples": int(fit_indices.size),
                "calibration_samples": int(cal_indices.size),
            }

        full = self._evaluate_mask(
            model, full_w, full_u, old_count, all_keep, X, T, fit_indices, cal_indices
        )
        ablations = []
        for removed in all_keep:
            item = self._evaluate_mask(
                model, full_w, full_u, old_count,
                [index for index in all_keep if index != removed],
                X, T, fit_indices, cal_indices,
            )
            item["removed"] = removed
            ablations.append(item)

        importance = {
            int(item["removed"]): float(item["loss"] - full["loss"])
            for item in ablations
        }
        order = sorted(importance, key=lambda index: (-importance[index], index))
        minimum = min(max(1, self.config.min_keep), proposed_count)
        candidates = [full]
        for count in range(minimum, proposed_count):
            candidates.append(
                self._evaluate_mask(
                    model, full_w, full_u, old_count, order[:count],
                    X, T, fit_indices, cal_indices,
                )
            )
        if self.config.loss_tolerance is None:
            chosen = min(
                candidates,
                key=lambda item: (
                    item["loss"] + self.config.sparsity_lambda * len(item["keep"]),
                    len(item["keep"]),
                    item["loss"],
                ),
            )
            gate_rule = "l0_penalized"
        else:
            eligible = [
                item for item in candidates
                if item["loss"] <= full["loss"] + self.config.loss_tolerance
            ]
            # The full block is always eligible for every nonnegative tolerance.
            chosen = min(eligible, key=lambda item: (len(item["keep"]), item["loss"]))
            gate_rule = "full_block_loss_constraint"
        self._set_mask(model, full_w, full_u, old_count, chosen["keep"])
        record = {
            "mode": "gatezo",
            "finite_difference": "one-sided hard coordinate ablation",
            "coordinate_queries": proposed_count,
            "proposed": proposed_count,
            "kept": len(chosen["keep"]),
            "keep": chosen["keep"],
            "importance": [importance[index] for index in all_keep],
            "importance_order": order,
            "full_loss": full["loss"],
            "chosen_loss": chosen["loss"],
            "chosen_accuracy": chosen["accuracy"],
            "gate_rule": gate_rule,
            "loss_tolerance": self.config.loss_tolerance,
            "chosen_objective": (
                chosen["loss"] + self.config.sparsity_lambda * len(chosen["keep"])
            ),
            "fit_samples": int(fit_indices.size),
            "calibration_samples": int(cal_indices.size),
        }
        return record


def evaluate_structure(model, X, T, fit_indices, validation_indices, ridge):
    fit_h, _ = model.getH(X[fit_indices])
    validation_h, _ = model.getH(X[validation_indices])
    beta = fit_readout(np.asarray(fit_h), T[fit_indices], ridge, _counter(model))
    return classification_metrics(np.asarray(validation_h), T[validation_indices], beta, _counter(model))


def refit_and_measure(model, X, T, X_test, T_test, ridge):
    train_h, _ = model.getH(X)
    beta = fit_readout(np.asarray(train_h), T, ridge, _counter(model))
    model.Beta = beta
    train_metrics = classification_metrics(np.asarray(train_h), T, beta, _counter(model))
    test_h, _ = model.getH(X_test)
    test_metrics = classification_metrics(np.asarray(test_h), T_test, beta, _counter(model))
    return train_metrics, test_metrics


def run_gatezo_growth(model, X, T, X_test, T_test, selector: StructuralGateZO):
    """Run the original BDNN growth protocol with either checkpoint source.

    ``legacy_test`` exactly follows the original selection policy: test accuracy
    saves the historical best state.  ``validation_checkpoint`` uses a held-out
    portion of training for the same purpose and touches test only after the
    best structure has been restored and its readout refitted on all training.
    """
    X, T = np.asarray(X), np.asarray(T)
    X_test, T_test = np.asarray(X_test), np.asarray(T_test)
    protocol = selector.config.growth_protocol
    if protocol == "validation_checkpoint":
        shared_runtime = getattr(model, '_pp3_runtime', None)
        growth_indices, checkpoint_indices = (shared_runtime.outer_array_indices
            if shared_runtime is not None else selector.make_outer_split(T))
        growth_X, growth_T = X[growth_indices], T[growth_indices]
        checkpoint_X, checkpoint_T = X[checkpoint_indices], T[checkpoint_indices]
    else:
        growth_X, growth_T = X, T
        checkpoint_X, checkpoint_T = X_test, T_test

    runtime = getattr(model, '_pp3_runtime', None)
    attempts = 0
    error_list, train_rates, test_rates, time_list = [], [], [], []
    rate_window = [0.0]
    previous_window_best = 0.0
    current_window_best = 1.0 / (
        5.0 if model.__class__.__name__.startswith("BDNN2") else 15.0
    )
    error = 30.0
    best_checkpoint_accuracy = -np.inf
    best_checkpoint_metrics = None
    best_state = None

    import time
    while (
        model.L < model.L_max
        and error > model.tol
        and (current_window_best - previous_window_best) >= 0.001
    ):
        if model.L != 0 and model.L % model.verbose == 0:
            previous_window_best = current_window_best
            current_window_best = max(rate_window)
            rate_window = []
        start = time.time()
        attempts += 1
        if runtime:
            runtime.begin('train', attempts, unknown='NumPy candidate scoring and non-matrix arithmetic not fully counted; readout linear algebra counted')
        old_count = 0 if isinstance(model.W, list) else int(np.asarray(model.W).shape[1])
        w_new, u_new, flag = model.sc_Search(growth_X, growth_T)
        if flag == 1:
            if runtime:
                runtime.end()
                runtime.epoch_row(attempts, unit='growth_round_failed')
            break
        model.addNodes(w_new, u_new)
        proposed = int(w_new.shape[1])
        record = selector.select(
            model, growth_X, growth_T, old_count, proposed
        )

        growth_h, _ = model.getH(growth_X)
        beta = fit_readout(np.asarray(growth_h), growth_T, selector.config.ridge, _counter(model))
        model.Beta = beta
        train_metrics = classification_metrics(
            np.asarray(growth_h), growth_T, beta, _counter(model)
        )
        checkpoint_h, _ = model.getH(checkpoint_X)
        checkpoint_metrics = classification_metrics(
            np.asarray(checkpoint_h), checkpoint_T, beta, _counter(model)
        )
        error = train_metrics["mse"]
        rate_window.append(train_metrics["accuracy"])
        if checkpoint_metrics["accuracy"] > best_checkpoint_accuracy:
            best_checkpoint_accuracy = checkpoint_metrics["accuracy"]
            best_checkpoint_metrics = checkpoint_metrics
            best_state = selector.snapshot(model)

        kept = record["kept"]
        elapsed = (time.time() - start) / 3600.0
        error_list.extend([error] * kept)
        train_rates.extend([train_metrics["accuracy"]] * kept)
        test_rates.extend([checkpoint_metrics["accuracy"]] * kept)
        time_list.extend([elapsed] * kept)
        record.update({
            "attempt": attempts,
            "status": "commit",
            "growth_round": int(model.L),
            "training_accuracy": train_metrics["accuracy"],
            "checkpoint_accuracy": checkpoint_metrics["accuracy"],
            "checkpoint_source": "test" if protocol == "legacy_test" else "validation",
            "hidden_after": int(np.asarray(model.W).shape[1]),
        })
        selector.history.append(record)
        if runtime:
            runtime.end(train_metrics['accuracy'], train_metrics['mse'])
            runtime.begin('test', attempts, unknown='NumPy non-matrix metric arithmetic not fully counted')
            test_h, _ = model.getH(X_test)
            observed = classification_metrics(np.asarray(test_h), T_test, model.Beta, _counter(model))
            runtime.end(observed['accuracy'], observed['mse'])
            runtime.epoch_row(attempts, unit='growth_round')
            runtime.rows[-1]['validation_accuracy'] = checkpoint_metrics['accuracy']
            runtime._csv()
        if selector.config.verbose:
            print(
                "[{}:{}] round={} proposed={} kept={} hidden={} "
                "train_acc={:.4f} checkpoint_acc={:.4f} mse={:.6f} time_h={:.6f}".format(
                    selector.config.selection_mode, protocol, attempts, proposed,
                    kept, record["hidden_after"], train_metrics["accuracy"],
                    checkpoint_metrics["accuracy"], error, elapsed,
                )
            )

    if best_state is None:
        raise RuntimeError("BDNN could not produce a checkpoint")
    if runtime:
        runtime.begin('final_refit_and_test', unknown='NumPy non-matrix evaluation arithmetic not fully counted')
    selector.restore(model, best_state)
    train_metrics, test_metrics = refit_and_measure(
        model, X, T, X_test, T_test, selector.config.ridge
    )
    if runtime:
        runtime.end(test_metrics['accuracy'], test_metrics['mse'])
    model.COST = train_metrics["mse"]
    model.gate_zo_history = selector.history
    model.gate_zo_summary = {
        "selection_mode": selector.config.selection_mode,
        "growth_protocol": protocol,
        "accepted_rounds": attempts,
        "attempted_rounds": attempts,
        "hidden_size": int(np.asarray(model.W).shape[1]),
        "structure_validation": (
            best_checkpoint_metrics
            if protocol == "validation_checkpoint" else None
        ),
        "training": train_metrics,
        "test": test_metrics,
        "checkpoint_accuracy": best_checkpoint_accuracy,
        "test_used_for_selection": protocol == "legacy_test",
    }
    return (
        np.asarray(error_list, dtype=np.float64).reshape(1, -1),
        np.asarray(train_rates, dtype=np.float64).reshape(1, -1),
        np.asarray(test_rates, dtype=np.float64).reshape(1, -1),
        np.asarray(time_list, dtype=np.float64).reshape(1, -1),
        clone_value(model.W), clone_value(model.U), clone_value(model.Beta),
    )
