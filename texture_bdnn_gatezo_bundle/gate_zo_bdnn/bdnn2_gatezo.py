"""Stage-2 incremental BDNN with structural gate-ZO growth."""

import numpy as np

from bdnn2 import bdnn2

from .cuda_backend import CudaBDNNMixin
from .structural_gate import GateZOConfig, StructuralGateZO, run_gatezo_growth


class BDNN2GateZO(CudaBDNNMixin, bdnn2):
    def __init__(
        self, *args, selection_mode="gatezo", growth_protocol="legacy_test",
        gate_zo_min_keep=1,
        gate_zo_sparsity_lambda=1e-4, gate_zo_loss_tolerance=None,
        adaptive_nozo_tolerance=1e-3,
        gate_zo_sample_size=None, gate_zo_calibration_ratio=0.5,
        gate_zo_ridge=0.0, gate_zo_validation_ratio=0.2,
        gate_zo_growth_min_delta=1e-4, gate_zo_failure_patience=10,
        gate_zo_seed=None, gate_zo_verbose=True,
        compute_device="auto", compute_batch_size=128, **kwargs
    ):
        kwargs["zo_steps"] = 0
        super().__init__(*args, **kwargs)
        self.configure_compute(compute_device, compute_batch_size)
        self.structural_gate = StructuralGateZO(GateZOConfig(
            selection_mode=selection_mode,
            growth_protocol=growth_protocol,
            min_keep=gate_zo_min_keep,
            sparsity_lambda=gate_zo_sparsity_lambda,
            loss_tolerance=gate_zo_loss_tolerance,
            adaptive_tolerance=adaptive_nozo_tolerance,
            sample_size=gate_zo_sample_size,
            calibration_ratio=gate_zo_calibration_ratio,
            ridge=gate_zo_ridge,
            validation_ratio=gate_zo_validation_ratio,
            growth_min_delta=gate_zo_growth_min_delta,
            failure_patience=gate_zo_failure_patience,
            seed=gate_zo_seed,
            verbose=gate_zo_verbose,
        ))

    def _load_and_filter_old_structure(self, X, T, w_path, u_path):
        self.W = np.load(w_path)
        self.U = np.load(u_path)
        hidden, _ = self.getH(X)
        selected = []
        r_value = self.r[-1]
        for index in range(hidden.shape[1]):
            values = []
            for output in range(T.shape[1]):
                values.append(self.inequalityEq(
                    T[:, output].reshape(-1, 1),
                    hidden[:, index].reshape(-1, 1),
                    r_value,
                ))
            if np.min(values) > 0:
                selected.append(index)
        # Do not destroy a valid pretrained model if the legacy inequality
        # happens to reject every old neuron on a small split.
        if selected:
            self.W = self.W[:, selected]
            self.U = self.U[np.ix_(selected, selected)]
        # Original bdnn2 counts only newly grown rounds and starts from zero.
        self.L = 0
        print("[gatezo] active pretrained neurons:", selected or "all")

    def classification(self, X, T, X2, T2, w_path, u_path):
        X, T = np.asarray(X), np.asarray(T)
        self._load_and_filter_old_structure(X, T, w_path, u_path)
        return run_gatezo_growth(self, X, T, X2, T2, self.structural_gate)


bdnn2_gatezo = BDNN2GateZO
