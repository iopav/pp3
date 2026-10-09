BDNN NumPy FLOP accounting
=========================

Scope and ownership
-------------------

``numpy_flops.py`` supplements the original ``bdnn1.py`` entry only. It does
not install global NumPy hooks, migrate the solver, or change CPU/GPU placement.
Calls outside an active runtime counter, disabled calls, and paused calls do
not add records. Each executed call contributes to its current runtime phase;
early exits do not incur hypothetical later candidate-search costs.

Torch operations still belong exclusively to ``TorchDispatchMode``. In
particular, Tensor * ndarray sometimes performs NumPy arithmetic and merely
wraps the result as a Tensor. The single-operation ``multiply`` boundary checks
the change in ATen arithmetic totals and unknown-call counts. It adds the
NumPy multiplication only if neither changed. Tensor/Tensor multiplication
is therefore counted once by ATen. Metrics use the same ownership check and
retain an unknown marker if a tensor arithmetic backend is observed.

Conversions, views, allocation, comparisons, indexing, sorting and integer
bookkeeping are excluded from floating arithmetic. Random-bit/uniform-variate
generation is excluded as an RNG-engine operation. The subsequent affine
transforms and the normal-distribution transform are included. The sklearn
metric formulas describe their numerical core, excluding input-validation
work. These boundaries are explicit in the individual operation rules.

Formulas
--------

* Real matrix product (m,k) @ (k,n): 2*m*k*n, matching the Torch MAC=2 convention.
* Candidate array 2*x-1: 2*N; Lambda scaling: N, with dispatch ownership checked.
* Candidate inequality on length N columns: three dots = 6*N; scalar
  subtraction and vector scaling = 1+N; square/divide/subtract = 3. Total
  7*N+4, split across ATen and NumPy as actually executed.
* Candidate score reduction: input elements minus output elements. The native
  ``sum(axis=0)`` on shape (1, classes) costs zero additions, not classes-1.
* Hidden/time averaging: one division per element after conversion to NumPy.
* Row softmax on shape (N,C): subtract, exp, sum and divide = 4*N*C-N.
* Unweighted MSE with uniform output averaging: 3*N*C+C.
* Unweighted accuracy: N equivalent FLOPs for the mean of N correctness flags.
* Confusion-matrix percentages: two floating operations per cell; the sum of
  integer counts and the unnormalized confusion matrix are integer bookkeeping.

Estimates (included in other_flops_estimated)
--------------------------------------------

``pinv`` is executed exactly once using the original NumPy function. For a real
2-D (m,n) input let p=max(m,n), q=min(m,n). The declared thin-SVD work model is
4*p*q^2+8*q^3. Add 1 for the singular-value cutoff, up to q reciprocals, m*q
scaling multiplies and 2*m*n*q reconstruction FLOPs. This is a shape-based
estimate, not an exact LAPACK gesdd count or a guaranteed bound; reciprocal
work is rank-dependent and SVD backend paths differ. Empty inputs cost zero.
The following pseudoinverse @ targets is a separate matrix-product record.
Complex or batched inputs keep an unknown marker rather than using this rule.

Legacy ``np.random.randn`` uses polar Box-Muller and caches the second sample.
The wrapper reads the pre-call cache flag without advancing the random stream.
For K newly generated pairs it records ceil(K*(28/pi+6)): seven arithmetic
operations per rejection proposal, expected 4/pi proposals per accepted pair,
then six operations to transform that pair. Actual rejection counts are not
observed, so this is an expected-cost estimate. Cached samples cost no new
transform. This rule applies to legacy RandomState, not Generator's Ziggurat.

Sources for algorithm structure (the SVD cost model above is our convention,
not a FLOP count promised by NumPy):

* https://numpy.org/doc/2.2/reference/generated/numpy.linalg.pinv.html
* https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html
* https://github.com/numpy/numpy/blob/main/numpy/random/src/legacy/legacy-distributions.c

Remaining coverage and old results
----------------------------------

Following the comparison scope selected on 2026-10-09, the train phase explicitly
excludes Python scalar loop/control arithmetic instead of marking it unknown.
Existing tensor arithmetic and NumPy formulas remain included. A scope-complete
result can therefore be numeric without certifying whole-program accounting.
Unknown Torch operators and other
unsupported paths still fail coverage. It does not repair unrelated BDNN
selection logic or add NumPy accounting to GateZO/ESPP/other methods.

Historical result files are not rewritten: the old files do not contain enough
information to reconstruct all executed NumPy calls. Rerun in a new output
directory to obtain the supplemented operation records; do not interpret an
existing completed-task cache as a measurement with this accounting version.

Validation
----------

``tests/test_bdnn_numpy_accounting.py`` checks independent small arithmetic
examples, mixed NumPy/Torch ownership, one-call pinv behavior, rank-deficient
and empty matrices, RNG cache/state preservation, early-exit candidate counts,
enabled/disabled/paused behavior, and counted versus uncounted native-entry
model/metric equality and phase totals. Existing FLOP regression tests remain
applicable. On the local Windows conda installation the tests use
``MKL_THREADING_LAYER=SEQUENTIAL`` to avoid duplicate OpenMP runtime loading;
``python_deps`` supplies the existing optional h5py dependency for the full suite.
