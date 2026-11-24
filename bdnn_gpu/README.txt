PP3 GPU BDNN (additive backend, native algorithm semantics)
=========================================================

Submit from the existing pp3 project directory on the cluster:

  sbatch submit_experiments_bdnn_gpu.sbatch --methods bdnn

Run the complete main experiment with only BDNN redirected to CUDA:

  sbatch submit_experiments_bdnn_gpu.sbatch

The new sbatch preserves the original account, partition, four-GPU allocation,
conda avalanche environment, worker/thread settings and inherited GPU tokens.
Data paths, seeds, model settings and representations come from constants.py.
The default output is /home/yizhil/pp3/result_bdnn_gpu (the original output
plus _bdnn_gpu). Override using --output. No original source file is edited.

Direct launch and summary:

  python run_experiments_bdnn_gpu.py --methods bdnn --datasets spike texture shd
  python summarize_experiments.py --inputs /home/yizhil/pp3/result_bdnn_gpu --output /home/yizhil/pp3/result_bdnn_gpu/summary.csv

The new entry uses the original controller itself, with a process-local entry
override for bdnn only. All other methods keep their original entries. Backend
identity differs from CPU BDNN so completed CPU results are not reused.

Preserved algorithm behavior
----------------------------
The code ports BDNNversion1-main/bdnn1.py to CUDA: delayed recurrent spikes,
candidate inequality, candidate counting by len(C), ascending candidate sort
with reversed class rows, accepted-U indexing, block growth, unchanged E=T,
pinv readout, separate filtered training loss and average-spike inference,
7-round accuracy stopping check (the native main entry's config override),
native first-history-entry length, test
argmax model selection and its exclusive final slice are all retained.
Consequently a one-round smoke run truncates to ZERO final nodes, exactly as
the original code does. No correction or fallback node is inserted. Empty
candidate histories still fail at the original final max operation.

Model arithmetic, recurrence, candidate scores, SVD/pinv, training loss and
inference run on CUDA. Data loading, NumPy random sampling and NumPy candidate
argsort stay on the host to retain the native random stream and tie ordering.
Random affine transforms are CUDA operations. Final CPU copies serialize the
checkpoint and do not participate in training. GPU arithmetic/SVD may differ
numerically from CPU; threshold or near-singular cases can amplify rounding.
Identical seeds do not establish bitwise CPU/GPU equivalence.

Spike/Texture keep their original whole-array execution unchanged. SHD uses
sequential input batches of exactly configured batch_size (default 128; only
the last batch may be smaller), without full dense input allocation on RAM/CUDA.
Time bins and augmentation are unchanged. Deterministic binary input frames
are packed to one bit per value on first access and reused from temporary
files under the run output directory. Subsequent passes do not repeat HDF5
event reads, jitter or binning. Initial cache creation is sequential.
Candidate scoring and pinv still operate on all N x hidden features together.
Training loss reuses exact binary spike histories from a temporary packed
cache after the global readout fit, matching native upgradeSCN's single
hidden pass. Cache I/O is included in timing/energy, excluded from arithmetic
FLOPs as data representation work. Temporary caches are removed normally;
SIGKILL can leave them in the failed run directory. Input cache needs roughly
N*T*C/8 bytes on disk, spike cache N*T*H/8, plus file headers. Memory scales
as O(chunk_size * time * (inputs + hidden + classes) + N * hidden), rather
than retaining N * time input/spike tensors. This is not minibatch fitting.
Floating-point kernel shape changes can affect near-threshold decisions;
bitwise equivalence is not guaranteed. No CPU fallback is used.

Only rerun failed SHD seeds; existing Spike/Texture results remain valid.
Use --datasets shd explicitly: source hashes change with this update, so the
controller may rerun other datasets if they are included in the same command.

Metrics
-------
metrics.csv contains the existing train/test/epoch FLOPs, accuracy, loss,
synchronized seconds and GPU joules, plus final_model_bytes, matching the
shared PP3 metrics schema (19 columns for BDNN).
GPU peak memory and train_stage_*_flops columns are not exported.
status.json retains final_refit as a one-time cost, final test accuracy, CUDA
device provenance and test_used_for_selection=true. The original summarizer
exports epoch costs and final_refit once each.

Per-phase epoch_XXXX_PHASE_operations.json files are not written.
flops_unknown.json retains the shared runtime's unknown-operation diagnostics.
MAC=2; scalar special functions=1; integer/control/copy operations follow the
existing PP3 convention. Pseudoinverse has the same shape-based work estimate
as native PP3: with q=min(m,n), p=max(m,n), its cost is
4*p*q*q + 8*q^3 + 1 + q + q*m + 2*n*q*m (zero for q=0).
Dispatch is paused only around pinv, then its estimate is added once. The
readout multiply is independently dispatched. Gaussian sampling retains the
native expected polar-transform estimate. These are arithmetic equivalents,
not hardware instruction counts. Unknown operations invalidate coverage.
Missing NVML telemetry remains blank; no fabricated energy is substituted.

final_model_bytes and the
summary's mem_bytes are serialized model.pt size, including serialization
overhead. They are not parameter bytes or VRAM. Auxiliary memory is persistent
optimizer plus hidden state at a completed growth-round boundary; both are
zero for this native non-BP model. Transient syn/mem/spikes are excluded from
persistent recovery state. No new resume protocol is
introduced; shared random state/data partitions are outside auxiliary memory.

Validation
----------
  python -m unittest discover -s tests -p test_bdnn_cuda_backend.py -v
  python -m unittest discover -s tests -p test_bdnn_chunked.py -v
  python -m bdnn_gpu.verify_integration

The integration command uses explicitly synthetic Spike/Texture arrays and SHD events and
the real PP3 launcher, loader, CUDA model, runtime, completion gate and summary.
Its artifacts live in outputs/bdnn_gpu_verification. It checks complete FLOPs,
train+test=epoch, CUDA placement and native zero-node final
slice, the shared metrics schema and absence of per-phase JSON files. The unit suite
also compares original and CUDA entries across the 7-round stopping checks.
These smoke results are not benchmark accuracies or convergence claims.
Formal dataset runs are to be executed with the above cluster sbatch command.
