import numpy as np

from bdnn1 import bdnn1
from gate_zo_bdnn.bdnn1_gatezo import BDNN1GateZO
from gate_zo_bdnn.structural_gate import (
    GateZOConfig, StructuralGateZO, cross_entropy, fit_readout, stratified_split,
)


def test_stratified_split_is_disjoint_and_covers_samples():
    targets = np.eye(3)[np.repeat(np.arange(3), 5)]
    fit, validation = stratified_split(targets, 0.4, np.random.default_rng(7))
    assert not set(fit).intersection(validation)
    assert sorted(np.concatenate([fit, validation]).tolist()) == list(range(15))
    assert set(np.argmax(targets[validation], axis=1)) == {0, 1, 2}


def test_ridge_readout_and_cross_entropy_are_finite():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]])
    targets = np.eye(2)[[0, 1, 0, 1]]
    beta = fit_readout(features, targets, 1e-6)
    assert beta.shape == (2, 2)
    assert np.isfinite(cross_entropy(features @ beta, targets))


def test_config_rejects_invalid_keep_count():
    try:
        GateZOConfig(min_keep=0).validate()
    except ValueError:
        return
    raise AssertionError("invalid min_keep was accepted")


def test_gate_zo_hard_ablation_commits_a_valid_substructure():
    class TinyModel:
        def __init__(self):
            self.W = np.asarray([[2.0, 0.1, -0.1], [0.0, 0.1, 0.1]])
            self.U = np.zeros((3, 3))
            self.Beta = []
            self.L = 3

        def getH(self, X):
            features = np.asarray(X).mean(axis=1) @ np.asarray(self.W)
            return features, np.zeros((X.shape[0], X.shape[1], self.W.shape[1]))

    rng = np.random.default_rng(3)
    class0 = rng.normal(loc=[-1.0, 0.0], scale=0.1, size=(12, 4, 2))
    class1 = rng.normal(loc=[1.0, 0.0], scale=0.1, size=(12, 4, 2))
    X = np.concatenate([class0, class1])
    T = np.eye(2)[[0] * 12 + [1] * 12]
    model = TinyModel()
    selector = StructuralGateZO(GateZOConfig(
        min_keep=1, sparsity_lambda=0.01, seed=5,
    ))
    record = selector.select(model, X, T, old_count=0, proposed_count=3)
    assert record["mode"] == "gatezo"
    assert 1 <= record["kept"] <= 3
    assert model.W.shape == (2, record["kept"])
    assert model.U.shape == (record["kept"], record["kept"])
    assert len(record["importance"]) == 3


def test_constrained_gate_respects_full_block_loss_tolerance():
    class TinyModel:
        def __init__(self):
            self.W = np.asarray([[2.0, 0.2, 0.1], [0.0, 0.1, 0.1]])
            self.U = np.zeros((3, 3))
            self.Beta = []
            self.L = 3

        def getH(self, X):
            features = np.asarray(X).mean(axis=1) @ np.asarray(self.W)
            return features, np.zeros((X.shape[0], X.shape[1], self.W.shape[1]))

    rng = np.random.default_rng(21)
    X = np.concatenate([
        rng.normal([-1.0, 0.0], 0.1, (12, 4, 2)),
        rng.normal([1.0, 0.0], 0.1, (12, 4, 2)),
    ])
    T = np.eye(2)[[0] * 12 + [1] * 12]
    selector = StructuralGateZO(GateZOConfig(
        min_keep=1, loss_tolerance=0.0, seed=8,
    ))
    record = selector.select(TinyModel(), X, T, 0, 3)
    assert record["gate_rule"] == "full_block_loss_constraint"
    assert record["loss_tolerance"] == 0.0
    assert record["chosen_loss"] <= record["full_loss"] + 1e-12


def test_fixed_nozo_keeps_the_complete_block_without_queries():
    class Model:
        W = np.zeros((2, 3))
        U = np.zeros((3, 3))
        Beta = []
        L = 3

    model = Model()
    selector = StructuralGateZO(GateZOConfig(selection_mode="fixed_nozo"))
    record = selector.select(
        model, np.zeros((6, 2, 2)), np.eye(2)[[0, 0, 0, 1, 1, 1]], 0, 3,
    )
    assert record["kept"] == 3
    assert record["coordinate_queries"] == 0


def test_adaptive_nozo_selects_a_prefix_without_coordinate_queries():
    class TinyModel:
        def __init__(self):
            self.W = np.asarray([[2.0, 0.2, 0.1], [0.0, 0.1, 0.1]])
            self.U = np.zeros((3, 3))
            self.Beta = []
            self.L = 3

        def getH(self, X):
            features = np.asarray(X).mean(axis=1) @ np.asarray(self.W)
            return features, np.zeros((X.shape[0], X.shape[1], self.W.shape[1]))

    rng = np.random.default_rng(9)
    X = np.concatenate([
        rng.normal([-1.0, 0.0], 0.1, (12, 4, 2)),
        rng.normal([1.0, 0.0], 0.1, (12, 4, 2)),
    ])
    T = np.eye(2)[[0] * 12 + [1] * 12]
    model = TinyModel()
    selector = StructuralGateZO(GateZOConfig(
        selection_mode="adaptive_nozo", min_keep=1,
        adaptive_tolerance=0.01, seed=4,
    ))
    record = selector.select(model, X, T, 0, 3)
    assert 1 <= record["kept"] <= 3
    assert record["coordinate_queries"] == 0
    assert record["prefix_evaluations"] == 3


def test_new_compute_backend_matches_legacy_cpu_hidden_rates():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(5, 9, 2)).astype(np.float32)
    W = rng.normal(scale=0.2, size=(2, 3)).astype(np.float32)
    U = rng.normal(scale=0.1, size=(3, 3)).astype(np.float32)
    legacy = bdnn1(alpha=0.13, beta=1.0)
    accelerated = BDNN1GateZO(
        alpha=0.13, beta=1.0, compute_device="cpu",
        compute_batch_size=2, gate_zo_verbose=False,
    )
    legacy.W, legacy.U = W.copy(), U.copy()
    accelerated.W, accelerated.U = W.copy(), U.copy()
    expected, _ = legacy.getH(X)
    actual, spikes = accelerated.getH(X)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert spikes is None
