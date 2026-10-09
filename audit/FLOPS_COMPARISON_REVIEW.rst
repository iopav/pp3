FLOPs comparison review, 2026-10-09
=================================

Decision
--------

Use 1 real MAC = 2 FLOPs. Exclude isolated Python scalar bookkeeping, retaining
existing tensor arithmetic. This is an explicit experimental scope, not exact
hardware instruction counting. Exclusions are recorded in operation summaries.
Major matrix products, convolution, repeated candidate evaluation, FFT and
linear algebra solvers must remain in scope, regardless of CPU/GPU placement.

Current operation inventory
---------------------------

* mm, bmm, matmul, dot, vdot: real dense MAC=2 (complex MAC=8 real arithmetic
  equivalents). Broadcasting uses the actual output shape. addmm/baddbmm also
  include bias/addend and non-unit alpha/beta scale arithmetic.
* mv/addmv: added in this review; matrix-vector MACs were previously unknown.
* Linear/einsum: counted through dispatched leaf operators, not by separately
  adding a module-level estimate. Autograd matrix products are counted as they
  execute; there is no extra global "training = 3 x forward" multiplier.
* Non-transposed convolution: 2*output_elements*(Cin/groups)*kernel_volume,
  plus bias if present. Backward charges the dense product estimate for each
  requested input/weight gradient and a bias reduction when requested.
  This is the conventional padded dense estimate, not a cuDNN algorithm trace.
* FFT: c2c coefficient 5, real transforms 2.5, times transformed volume,
  batch count and sum(log2(axis_lengths)); normalization is included. Estimated,
  especially for non-power-of-two sizes; must not be dropped for SPSN.
* Real unbatched left linear solve: (2/3)*n^3 + 2*n^2*rhs_columns estimate.
  Batched/complex/right solves remain unknown.
* BDNN real 2-D NumPy pinv: declared SVD estimate plus reconstruction; subsequent
  product with targets counted separately. Not an exact LAPACK operation count.
* Elementwise arithmetic, scaled/fused arithmetic, reductions, vector norms,
  softmax/losses, activation arithmetic, batchnorm, foreach optimizer arithmetic,
  indexed accumulation and gradient scatter are included where a rule exists.
* BDNN NumPy candidate scores, averaging, softmax, MSE/accuracy and affine
  candidate generation are manually supplemented. Mixed Tensor/ndarray
  multiplication checks ATen ownership before any manual contribution.
* Copy/view/indexing/comparison/sorting/integer bookkeeping are excluded.
  ReLU and hard thresholds cost zero under the existing comparison convention.
* Special scalar functions cost one equivalent; this is neither their latency
  nor a hardware instruction count. Random-bit engine costs are excluded;
  BDNN normal transforms have an expected-cost estimate. Torch normal RNG
  transforms remain unknown where encountered.

Checks performed
----------------

Independent arithmetic examples: mm (2,3)@(3,4)=48; bmm with five such products
=240; the same linear layer with bias=56; forward plus both matrix gradients
=144; grouped convolution with input (2,4,5,5), weights (6,2,3,3), groups=2
=3888; rFFT of (2,8)=120 under the FFT model; solve (3,3) with two RHS=54
under the LU model. mv (2,3)@(3,) previously returned unknown, now gives 12.
addmv checks cover default bias, non-unit alpha/beta and beta=0.

BDNN has dedicated numerical/transparency tests including one-call pinv,
rank-deficient/empty inputs, RNG cache preservation, actual search-call counts,
and counted/uncounted native-entry model and metric equality. A separate
two-round comparison against the pre-supplement git source showed identical
ATen records; all added work was in NumPy records. This is smoke evidence, not
a rerun of all formal datasets or proof for every possible operator shape.

Remaining comparison blockers
-----------------------------

GateZO Spike/Texture uses the bundle structural_gate.py, not the SHD entry.
The subsequent GateZO supplement now instruments NumPy pinv, Gram-matrix
products, RHS products, solve, and prediction matrix products in both bundles'
fit_readout and classification_metrics. The stage-1 candidate scoring dot
products are also counted. Shared formulas live in pp3/numpy_linalg.py, use
MAC=2, and use the same pinv model as BDNN. Each mask evaluation, repeated
fit, validation/test and final refit receives the active phase counter.

Small non-matrix NumPy arithmetic (candidate transforms, nonlinear scoring,
ridge diagonal construction and metric arithmetic) is still explicitly marked
uncovered, rather than declaring whole-program completeness. A failed solve
retains an unknown partial-factorization cost; its pinv fallback and product
are counted once each. SHD and unused stage-2/continuous-ZO paths are unchanged.

GateZO SHD has a different Torch-based path; the implemented solve rule covers
only its documented real unbatched left-solve case. Its broad initialization
and proposal unknown markers require path-specific review, not blanket removal.

ESPP's current runtime calls train_out_proj_fast (Torch readout training), not
the separate unused train_out_proj_closed_form (SciPy/sklearn). It would be
incorrect to declare missing Ridge/SVD costs merely because that unused helper
exists. Existing broad readout/initialization unknown markers are retained;
scalar-only markers on later train phases are now explicit exclusions.

The other neural-network paths have matrix/conv/FFT accounting and no need to
chase Python Adam bias-correction or scheduler scalars. Old audit files predate
several rules (BN, SiLU, solve, BDNN NumPy); their unknown lists are not evidence
that the same current rules are missing. Fresh full-data records are required.

Reporting rules for an experiment
---------------------------------

Use the same samples, time steps, precision convention and measurement scope.
Dense SNN matmuls are charged as executed; spike sparsity does not automatically
turn them into sparse synaptic operations. Report train and evaluation costs
separately. Preparation/final refit are one-time costs and must be added once
when comparing total training work. A growth round is not equivalent to a
backpropagation epoch; compare cumulative cost to a stated accuracy target or
the total cost of each declared training protocol.

summary.csv currently falls back from missing flops to known_flops. Its numeric
value alone cannot certify comparison coverage; inspect status and operation
unknown records. Missing scalar bookkeeping and missing matrix solves have
different consequences. Existing results are not retroactively updated.

References for the convention
-----------------------------

NVIDIA's matrix-multiplication guide states MNK FMAs and 2MNK FLOPs:
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

PyTorch profiler documents formula-based estimates for matrix multiplication
and 2-D convolution (not exhaustive whole-program counting):
https://docs.pytorch.org/docs/stable/profiler
