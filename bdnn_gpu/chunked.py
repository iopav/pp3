"""SHD-only bounded input batches; full-data candidate scoring and pinv retained."""
from contextlib import nullcontext
import torch
from .model import BDNNGPU


class ChunkedInput:
    def __init__(self, dataset, device, batch_size=8, runtime=None):
        self.dataset, self.device = dataset, torch.device(device)
        self.batch_size, self.runtime = max(1, int(batch_size)), runtime
        sample, _ = dataset[0]
        self.shape = (len(dataset), *sample.shape)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        # Sequential indexing preserves native arrays() order and deterministic
        # augmentation; no workers/prefetch or dataset-sized dense allocation.
        for start in range(0, len(self), self.batch_size):
            context = self.runtime.preprocessing() if self.runtime else nullcontext()
            with context:
                batch = torch.stack([self.dataset[i][0] for i in
                    range(start, min(start + self.batch_size, len(self)))])
                batch = batch.to(self.device)
            yield batch
            del batch


class ChunkedBDNNGPU(BDNNGPU):
    def hidden(self, x, w, u):
        if not isinstance(x, ChunkedInput):
            return super().hidden(x, w, u)
        # Only N x hidden features survive each batch. Candidate scoring still
        # uses all samples together, in the original order.
        features = []
        for batch in x:
            h, spikes = super().hidden(batch, w, u)
            features.append(h)
            del spikes
        return torch.cat(features), None

    def upgradeSCN(self, x, target):
        if not isinstance(x, ChunkedInput):
            return super().upgradeSCN(x, target)
        h, _ = self.getH(x)
        self.computeBeta(h, target)
        # Recompute batch spike histories after the single global pinv fit.
        # This trades an extra recurrence pass for bounded memory. Its actual
        # FLOPs, elapsed time and energy remain inside the training phase.
        results = []
        for batch in x:
            _, spikes = super().hidden(batch, self.W, self.U)
            drive = torch.einsum('abc,cd->abd', spikes.float(), self.Beta.float())
            filt = batch.new_zeros((len(batch), target.shape[1]))
            output, history = filt.clone(), []
            for t in range(batch.shape[1]):
                filt, output = self.alpha * filt + drive[:, t], self.beta * output + filt
                history.append(output)
            results.append(self.softmax(torch.stack(history, 1).sum(1)))
            del spikes, drive, history
        result = torch.cat(results)
        self.COST = (target - result).square().mean(0).mean()
        return result, self.COST
