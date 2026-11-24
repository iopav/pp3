"""SHD-only bounded input batches; full-data candidate scoring and pinv retained."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import numpy as np
import torch
from .model import BDNNGPU


class ChunkedInput:
    def __init__(self, dataset, device, batch_size=128, runtime=None, cache_binary=False):
        self.dataset, self.device = dataset, torch.device(device)
        self.batch_size, self.runtime = int(batch_size), runtime
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        sample, _ = dataset[0]
        self.shape = (len(dataset), *sample.shape)
        self.cache = (tempfile.TemporaryDirectory(prefix='.bdnn_input_',
            dir=runtime.output if runtime else None) if cache_binary else None)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        # Sequential indexing preserves native arrays() order and deterministic
        # augmentation; no workers/prefetch or dataset-sized dense allocation.
        for start in range(0, len(self), self.batch_size):
            context = self.runtime.preprocessing() if self.runtime else nullcontext()
            with context:
                stop = min(start + self.batch_size, len(self))
                path = Path(self.cache.name)/f'{start}.npy' if self.cache else None
                shape = (stop - start, *self.shape[1:])
                if path is not None and path.exists():
                    batch = read_binary(path, shape, self.device)
                else:
                    batch = torch.stack([self.dataset[i][0] for i in range(start, stop)])
                    if path is not None:
                        write_binary(path, batch)
                batch = batch.to(self.device)
            yield batch
            del batch

    def close(self):
        if self.cache:
            self.cache.cleanup()


def write_binary(path, tensor):
    array = tensor.detach().cpu().numpy()
    if not np.all((array == 0) | (array == 1)):
        raise ValueError('Binary cache requires exact 0/1 values')
    np.save(path, np.packbits(array.astype(np.uint8).reshape(-1)), allow_pickle=False)


def read_binary(path, shape, device):
    packed = np.load(path, allow_pickle=False)
    # Decode one batch to bytes; convert it to float on the destination device.
    array = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


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
            del spikes, batch
        return torch.cat(features), None

    def upgradeSCN(self, x, target):
        if not isinstance(x, ChunkedInput):
            return super().upgradeSCN(x, target)
        # Like native upgradeSCN: run hidden once, fit globally, reuse spikes.
        # Store exact binary spikes on disk instead of retaining N*T*H floats.
        results, features, shapes = [], [], []
        with tempfile.TemporaryDirectory(prefix='.bdnn_spikes_',
                dir=x.runtime.output if x.runtime else None) as folder:
            for index, batch in enumerate(x):
                h, spikes = super().hidden(batch, self.W, self.U)
                features.append(h)
                shapes.append(tuple(spikes.shape))
                context = x.runtime.preprocessing() if x.runtime else nullcontext()
                with context:
                    write_binary(Path(folder)/f'{index}.npy', spikes)
                del spikes, batch
            self.computeBeta(torch.cat(features), target)
            for index, shape in enumerate(shapes):
                context = x.runtime.preprocessing() if x.runtime else nullcontext()
                with context:
                    spikes = read_binary(Path(folder)/f'{index}.npy', shape, self.device)
                drive = torch.einsum('abc,cd->abd', spikes, self.Beta.float())
                filt = drive.new_zeros((shape[0], target.shape[1]))
                output, history = filt.clone(), []
                for t in range(x.shape[1]):
                    filt, output = self.alpha * filt + drive[:, t], self.beta * output + filt
                    history.append(output)
                results.append(self.softmax(torch.stack(history, 1).sum(1)))
                del spikes, drive, history
        result = torch.cat(results)
        self.COST = (target - result).square().mean(0).mean()
        return result, self.COST
