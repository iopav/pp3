"""CUDA forward/search backend for the isolated BDNN variants.

The original BDNN files create every tensor on CPU.  This mixin preserves their
LIF equations and NumPy random candidate generation, while moving the expensive
time-unrolled simulations to one explicit Torch device.  Hidden spike counts
are copied back only after a full sample batch has finished.
"""

from __future__ import annotations

import numpy as np
import torch


class CudaBDNNMixin:
    def configure_compute(self, compute_device="auto", compute_batch_size=128):
        requested = str(compute_device).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False; "
                "run the container with apptainer exec --nv"
            )
        self.compute_device = torch.device(requested)
        self.compute_batch_size = max(1, int(compute_batch_size))
        print(
            "[compute] device={} batch_size={} cuda_available={}".format(
                self.compute_device, self.compute_batch_size,
                torch.cuda.is_available(),
            )
        )

    def _simulate_spike_rates(self, X, W, U):
        """Return mean spike counts using the original update ordering."""
        X = np.asarray(X, dtype=np.float32)
        W_device = torch.as_tensor(
            np.asarray(W), dtype=torch.float32, device=self.compute_device
        )
        U_device = torch.as_tensor(
            np.asarray(U), dtype=torch.float32, device=self.compute_device
        )
        hidden = W_device.shape[1]
        steps = X.shape[1]
        outputs = []
        with torch.inference_mode():
            for start in range(0, X.shape[0], self.compute_batch_size):
                stop = min(start + self.compute_batch_size, X.shape[0])
                batch = torch.as_tensor(
                    X[start:stop], dtype=torch.float32, device=self.compute_device
                )
                projected = torch.matmul(batch, W_device)
                syn = torch.zeros(
                    (stop - start, hidden), dtype=torch.float32,
                    device=self.compute_device,
                )
                mem = torch.zeros_like(syn)
                out = torch.zeros_like(syn)
                spike_sum = torch.zeros_like(syn)
                for step in range(steps):
                    current = projected[:, step, :] + torch.matmul(out, U_device)
                    out = (mem > 1.0).to(torch.float32)
                    new_syn = self.alpha * syn + current
                    new_mem = (self.beta * mem + syn) * (1.0 - out)
                    spike_sum.add_(out)
                    syn, mem = new_syn, new_mem
                outputs.append((spike_sum / steps).cpu())
                del batch, projected, syn, mem, out, spike_sum
        return torch.cat(outputs, dim=0).numpy()

    def getH(self, X):
        """CUDA-backed replacement used by all three comparison methods."""
        if isinstance(self.W, list) or isinstance(self.U, list):
            raise RuntimeError("getH requires at least one hidden neuron")
        hidden = self._simulate_spike_rates(X, self.W, self.U)
        # The new structural code consumes only the aggregate hidden features;
        # avoiding a [samples,time,hidden] spike tensor saves substantial memory.
        return hidden, None

    def sc_Search(self, X, E0):
        """CUDA-backed candidate search with scalar scores and source indices."""
        X = np.asarray(X)
        E0 = np.asarray(E0)
        dimensions = X.shape[2]
        output_count = E0.shape[1]
        candidates = []
        candidate_w = []
        candidate_u = []
        candidate_sources = []
        candidate_count = 0

        for pool_id, lambda_value in enumerate(self.Lambdas):
            scale = float(lambda_value.item())
            W_pool = scale * (2 * np.random.rand(dimensions, self.T_max) - 1)
            U_pool = scale * (2 * np.random.randn(self.T_max, self.T_max) - 1)
            hidden = self._simulate_spike_rates(X, W_pool, U_pool)

            for r_value in self.r:
                for index in range(self.T_max):
                    feature = hidden[:, index].reshape(-1, 1)
                    ksi = np.zeros((1, output_count), dtype=np.float64)
                    for output in range(output_count):
                        target = E0[:, output].reshape(-1, 1)
                        ksi[0, output] = self.inequalityEq(
                            target, feature, r_value
                        )
                    if np.min(ksi) > 0:
                        # Rank each candidate by its total over all outputs.
                        score = float(np.sum(ksi))
                        candidates.append(score)
                        candidate_w.append(W_pool[:, index].reshape(-1, 1))
                        candidate_u.append(U_pool[:, index].reshape(-1, 1))
                        candidate_sources.append((pool_id, index))
                candidate_count = len(candidates)
                if candidate_count >= self.nB:
                    break
            if candidate_count >= self.nB:
                break

        if not candidates or candidate_count < self.nB:
            return [[], [], 1]

        C = np.asarray(candidates)
        W_block = np.concatenate(candidate_w, axis=1)
        U_columns = np.concatenate(candidate_u, axis=1)
        selected = np.argsort(-C, kind="stable")[:self.nB]
        W_block = W_block[:, selected]
        # Rows of U_columns retain original pool indices; columns are filtered.
        # Independent pools have no sampled connections between them.
        sources = np.asarray(candidate_sources)[selected]
        U_block = U_columns[np.ix_(sources[:, 1], selected)].copy()
        U_block[sources[:, 0, None] != sources[None, :, 0]] = 0.0
        return [W_block, U_block, 0]
