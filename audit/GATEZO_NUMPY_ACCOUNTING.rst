GateZO Spike/Texture NumPy linear algebra supplement
==================================================

Both production bundles pass their active counter explicitly through structural
mask/prefix evaluations, readout fitting, predictions and final refitting.
Stage-1 candidate inequality dot products include the denominator-only early
exit. No global NumPy function is patched in production. Standalone calls,
disabled counters and paused counters retain numerical behavior without adding
counts. The Torch simulation and matrix-product code is untouched.

Real (m,k)@(k,n) costs 2*m*k*n. A square n-dimensional LU solve with r RHS
columns costs round(2*n^3/3+2*n^2*r), explicitly estimated. Real 2-D pinv uses
the same estimate as BDNN: p=max(m,n), q=min(m,n), cost
4*p*q^2+8*q^3+1+q+q*m+2*n*q*m; empty matrices cost zero. Solver formulas model
arithmetic work rather than actual LAPACK instruction counts.

A failed solve retains an unknown partial-factorization cost and preserves the
original pinv fallback. That fallback and its product are each counted once;
no solver is rerun for measurement. Unsupported operand types remain unknown,
and Tensor operations never receive NumPy formulas. Non-matrix NumPy arithmetic
is outside this supplement and its unknown markers remain with corrected
reasons. Historical results are not rewritten: fresh runs are needed.

tests/test_gatezo_numpy_accounting.py checks both bundles against independent
matrix/solver totals, empty inputs, paused/disabled counters and a forced solve
failure with exactly one fallback. Small native-entry runs compare model arrays,
prediction metrics and NumPy/Torch RNG states with counting on/off. Repeated
solver calls and phase totals are checked. The Spike native test exercises
pinv, Texture exercises ridge; unit cases exercise both in both bundles.
These are CPU smoke tests, not full-data or GPU formal experiments.
