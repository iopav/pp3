"""Explicit NumPy linear-algebra accounting; never also count Torch operands.

The pinv work model matches BDNNversion1-main/numpy_flops.py. Both are
shape-based estimates, not traces of LAPACK's actual backend instructions.
"""
import numpy as np


def active(counter):
    return counter is not None and counter.enabled and not counter.paused


def supported(counter, name, *arrays):
    if not active(counter):
        return False
    if all(isinstance(a, np.ndarray) and a.ndim == 2 and a.dtype.kind == 'f'
           for a in arrays):
        return True
    counter.mark_unknown('numpy::'+name, 'Only real floating 2-D NumPy operands audited')
    return False


def matmul(counter, left, right):
    result = left @ right
    if supported(counter, 'matmul', left, right):
        counter.add_manual('numpy::matmul', 2*left.shape[0]*left.shape[1]*right.shape[1],
                           rule='numpy_dense_matmul_mac_2')
    return result


def pinv(counter, matrix):
    result = np.linalg.pinv(matrix)
    if supported(counter, 'pinv', matrix):
        m, n = matrix.shape
        q, p = min(m, n), max(m, n)
        cost = 0 if q == 0 else 4*p*q*q + 8*q**3 + 1 + q + q*m + 2*n*q*m
        counter.add_manual('numpy::pinv', cost,
                           rule='numpy_real_pinv_thin_svd_4pq2_8q3_plus_reconstruction_estimate')
    return result


def solve(counter, matrix, rhs):
    try:
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        # A failed factorization consumed work but has no successful full solve.
        # Preserve the original fallback; do not rerun or fabricate its cost.
        if active(counter):
            counter.mark_unknown('numpy::solve_failed', 'Partial failed LU factorization cost unknown; fallback counted separately')
        raise
    if supported(counter, 'solve', matrix, rhs):
        n = matrix.shape[0]
        counter.add_manual('numpy::solve', round(2*n**3/3 + 2*n*n*rhs.shape[1]),
                           rule='numpy_dense_lu_factor_and_triangular_solve_leading_term_estimate')
    return result
