"""Stage-1 BDNN with one structural gate-ZO step after every growth round."""

from bdnn1 import bdnn1

from .cuda_backend import CudaBDNNMixin
from .structural_gate import GateZOConfig, StructuralGateZO, run_gatezo_growth


class BDNN1GateZO(CudaBDNNMixin, bdnn1):
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
        # Disable the old continuous-weight ZO even if it is present in kwargs.
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

    def classification(self, X, T, X2, T2):
        return run_gatezo_growth(self, X, T, X2, T2, self.structural_gate)


# Familiar lowercase alias without shadowing the original module.
bdnn1_gatezo = BDNN1GateZO
