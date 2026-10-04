"""Configuration for the isolated gate-ZO experiment."""

from config import SCN_DEFAULTS


GATE_ZO_DEFAULTS = dict(SCN_DEFAULTS)
GATE_ZO_DEFAULTS.update({
    "selection_mode": "gatezo",
    "growth_protocol": "legacy_test",
    # The legacy continuous-weight ZO is deliberately disabled.
    "zo_steps": 0,
    # With the current nB=3, gate-ZO may retain one, two, or all three nodes.
    "gate_zo_min_keep": 1,
    "gate_zo_sparsity_lambda": 1e-4,
    # None retains the original L0-penalized rule.  A nonnegative value uses
    # the smallest subset within this calibration-loss distance of the full block.
    "gate_zo_loss_tolerance": None,
    "adaptive_nozo_tolerance": 1e-3,
    # None evaluates the complete training-side split (482 samples in exp2).
    "gate_zo_sample_size": None,
    "gate_zo_calibration_ratio": 0.5,
    "gate_zo_validation_ratio": 0.2,
    "gate_zo_ridge": 0.0,
    "gate_zo_growth_min_delta": 1e-4,
    "gate_zo_failure_patience": 10,
    "gate_zo_verbose": True,
    "compute_device": "auto",
    "compute_batch_size": 128,
})
