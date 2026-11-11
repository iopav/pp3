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
50-round accuracy stopping check, native first-history-entry length, test
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

Whole-array execution and spike-history retention follow the original code.
This intentionally does not introduce minibatch training or memory-saving
algorithm changes. Full SHD may exceed available VRAM; no CPU fallback is used.

Metrics
-------
metrics.csv contains the existing train/test/epoch FLOPs, accuracy, loss,
synchronized seconds and GPU joules, plus parameter tensor bytes, optimizer
state bytes, auxiliary state bytes and phase peak CUDA allocated/reserved
bytes. train_stage_*_flops provides search/readout-loss/train-evaluation work.
status.json retains final_refit as a one-time cost, final test accuracy, CUDA
device provenance and test_used_for_selection=true. The original summarizer
exports epoch costs and final_refit once each.

epoch_XXXX_PHASE_operations.json records operator rules, calls, phase totals,
substage totals and unknown operations. flops_unknown.json is also retained.
MAC=2; scalar special functions=1; integer/control/copy operations follow the
existing PP3 convention. Pseudoinverse has the same shape-based work estimate
as native PP3: with q=min(m,n), p=max(m,n), its cost is
4*p*q*q + 8*q^3 + 1 + q + q*m + 2*n*q*m (zero for q=0).
Dispatch is paused only around pinv, then its estimate is added once. The
readout multiply is independently dispatched. Gaussian sampling retains the
native expected polar-transform estimate. These are arithmetic equivalents,
not hardware instruction counts. Unknown operations invalidate coverage.
Missing NVML telemetry remains blank; no fabricated energy is substituted.

para_mem_bytes counts actual tensor dtype bytes. final_model_bytes and the
summary's mem_bytes are serialized model.pt size, including serialization
overhead. They are not parameter bytes or VRAM. Auxiliary memory is persistent
optimizer plus hidden state at a completed growth-round boundary; both are
zero for this native non-BP model. Transient syn/mem/spikes remain represented
by peak CUDA memory, not persistent recovery state. No new resume protocol is
introduced; shared random state/data partitions are outside auxiliary memory.

Validation
----------
  python -m unittest discover -s tests -p test_bdnn_cuda_backend.py -v
  python -m bdnn_gpu.verify_integration

The integration command uses explicitly synthetic Spike/Texture arrays and
the real PP3 launcher, loader, CUDA model, runtime, completion gate and summary.
Its artifacts live in outputs/bdnn_gpu_verification. It checks complete FLOPs,
train+test=epoch, stage sums=train, CUDA placement and native zero-node final
slice. These smoke results are not benchmark accuracies or convergence claims.
Formal dataset runs are to be executed with the above cluster sbatch command.
