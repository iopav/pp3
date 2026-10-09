"""Explicit accounting at BDNN's NumPy boundaries; no global NumPy hooks.

Torch arithmetic remains owned by TorchDispatchMode. These helpers never rerun
an operation to measure it. Matrix products use MAC=2, scalar functions cost 1.
SVD and legacy Gaussian transforms are estimates, not BLAS instruction counts.
"""
import math

import numpy as np
import torch


def active(counter):
    return counter is not None and counter.enabled and not counter.paused


def record(counter, name, flops, rule='numpy_arithmetic_equivalent'):
    if active(counter):
        counter.add_manual('numpy::' + name, int(flops), rule=rule)


def real_arrays(counter, name, *values):
    """Reject tensor/mixed or complex formulas rather than silently double count."""
    if not active(counter):
        return False
    if all(isinstance(v, np.ndarray) and v.dtype.kind == 'f' for v in values):
        return True
    counter.mark_unknown('numpy::' + name, 'Expected real floating NumPy arrays')
    return False


def dispatch_stamp(counter):
    rows = [r for r in counter.records.values() if not r['operator'].startswith('manual::')]
    return (sum(r['flops'] for r in rows),
            sum(r['calls'] for r in rows if r['flops_per_call'] is None))


def multiply(counter, scalar, array):
    # Tensor * ndarray can do NumPy arithmetic then wrap a Tensor result,
    # WITHOUT aten.mul. Output type alone cannot establish ownership.
    before = dispatch_stamp(counter) if active(counter) else None
    result = scalar * array
    if active(counter) and dispatch_stamp(counter) == before:
        if real_arrays(counter, 'mixed_scale', array):
            if isinstance(scalar, torch.Tensor) and (scalar.numel() != 1 or scalar.is_complex()):
                counter.mark_unknown('numpy::mixed_scale', 'Only a real scalar scale is audited')
            elif isinstance(scalar, complex):
                counter.mark_unknown('numpy::mixed_scale', 'Complex scale not audited')
            else:
                record(counter, 'mixed_scale', array.size,
                       'numpy_scalar_multiply_no_aten_arithmetic_observed')
    return result


def randn(counter, *shape):
    # NumPy's legacy RandomState uses the polar Box-Muller method and caches
    # the second variate. Inspecting this flag does not advance the RNG.
    cached = np.random.get_state()[3] if active(counter) else 0
    result = np.random.randn(*shape)
    if active(counter):
        pairs = (max(0, result.size - cached) + 1) // 2
        # Each proposal: 4 affine ops + 2 squares + 1 add = 7.
        # Acceptance pi/4 gives 28/pi expected ops per accepted pair.
        # Accepted pair: log, multiply, divide, sqrt, two multiplies = 6.
        record(counter, 'randn_transform', math.ceil(pairs * (28 / math.pi + 6)),
               'numpy_legacy_polar_expected_cost_estimate_rng_engine_excluded')
    return result


def pinv(counter, matrix):
    result = np.linalg.pinv(matrix)
    if real_arrays(counter, 'pinv', matrix):
        if matrix.ndim != 2:
            counter.mark_unknown('numpy::pinv', 'Only the BDNN real 2-D default pinv is audited')
        else:
            m, n = matrix.shape
            q, p = min(m, n), max(m, n)
            # Deliberately declared dense thin-SVD cost MODEL, not an exact
            # gesdd count: 4*p*q^2 + 8*q^3. Add cutoff, at most q reciprocals,
            # scaling U.T (q*m), and V @ scaled_U.T (2*n*q*m).
            # Rank-dependent reciprocals and backend branches stay estimated.
            cost = 0 if q == 0 else 4*p*q*q + 8*q**3 + 1 + q + q*m + 2*n*q*m
            record(counter, 'pinv', cost,
                   'numpy_real_pinv_thin_svd_4pq2_8q3_plus_reconstruction_estimate')
    return result


def matmul(counter, left, right):
    result = left @ right
    # A Tensor operand may own this operation through ATen. Never add a
    # NumPy formula to that mixed case without an independent audit.
    if real_arrays(counter, 'matmul', left, right):
        if left.ndim == right.ndim == 2:
            record(counter, 'matmul', 2 * left.shape[0] * left.shape[1] * right.shape[1],
                   'numpy_dense_matmul_mac_2')
        else:
            counter.mark_unknown('numpy::matmul', 'Only real 2-D matmul is audited')
    return result


def softmax(counter, function, matrix):
    result = function(matrix)
    if real_arrays(counter, 'softmax', matrix) and matrix.ndim == 2:
        # subtract max, exp, row sum, divide. Comparisons are excluded.
        record(counter, 'softmax', 4*matrix.size - matrix.shape[0])
    return result


def metric(counter, name, function, *args):
    # sklearn may select an array backend. If it dispatches any Torch work,
    # decline the aggregate formula instead of adding it over the ATen rows.
    before = dispatch_stamp(counter) if active(counter) else None
    result = function(*args)
    if active(counter):
        after = dispatch_stamp(counter)
        if after != before or isinstance(result, torch.Tensor):
            counter.mark_unknown('numpy::' + name, 'Metric dispatched tensor operations; aggregate not added')
        elif name == 'mean_squared_error' and real_arrays(counter, name, *args):
            n, c = (args[0].shape if args[0].ndim == 2 else (args[0].size, 1))
            record(counter, name, 3*n*c + c,
                   'numpy_unweighted_mse_sub_square_two_means_validation_excluded')
        elif name == 'accuracy_score':
            record(counter, name, len(args[0]),
                   'numpy_unweighted_accuracy_float_mean_comparisons_validation_excluded')
        else:
            counter.mark_unknown('numpy::' + name, 'Unsupported metric input or options')
    return result
