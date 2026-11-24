"""SHD-only bounded input batches; full-data candidate scoring and pinv retained."""
from contextlib import nullcontext
import time
import torch
from .model import BDNNGPU
from .binary_cache import BinaryCache, CacheBudget


class ChunkedInput:
    def __init__(self, dataset, device, batch_size=128, runtime=None, cache_binary=False):
        self.dataset, self.device = dataset, torch.device(device)
        self.batch_size, self.runtime = int(batch_size), runtime
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        sample, _ = dataset[0]
        self.shape = (len(dataset), *sample.shape)
        self.budget = getattr(runtime, 'binary_cache_budget', None) or CacheBudget()
        self.cache = BinaryCache(self.budget, runtime.output if runtime else None) if cache_binary else None

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        # Sequential indexing preserves native arrays() order and deterministic
        # augmentation; no workers/prefetch or dataset-sized dense allocation.
        for start in range(0, len(self), self.batch_size):
            context = self.runtime.preprocessing() if self.runtime else nullcontext()
            with context:
                stop = min(start + self.batch_size, len(self))
                shape = (stop - start, *self.shape[1:])
                if self.cache is not None and start in self.cache:
                    batch = self.cache.get(start, shape, self.device)
                else:
                    batch = torch.stack([self.dataset[i][0] for i in range(start, stop)])
                    if self.cache is not None:
                        self.cache.put(start, batch)
                batch = batch.to(self.device)
            yield batch
            del batch

    def close(self):
        if self.cache:
            self.cache.close()

    def progress(self, phase, done, tick, force=False):
        if self.runtime and (force or time.monotonic() - tick >= 30):
            print(f'round={self.runtime.epoch} phase={self.runtime.phase} {phase} '
                  f'samples={done}/{len(self)} cache_ram_gib={self.budget.used / 1024**3:.2f}', flush=True)
            return time.monotonic()
        return tick


class ChunkedBDNNGPU(BDNNGPU):
    def sc_Search(self, x, target, candidates=None):
        self._candidate_pass = 0
        self._searching = True
        try:
            return super().sc_Search(x, target, candidates)
        finally:
            self._searching = False

    def hidden(self, x, w, u):
        if not isinstance(x, ChunkedInput):
            return super().hidden(x, w, u)
        # Only N x hidden features survive each batch. Candidate scoring still
        # uses all samples together, in the original order.
        features = []
        done, tick = 0, time.monotonic()
        label = 'hidden'
        if getattr(self, '_searching', False):
            scale = self.settings['lambdas'][self._candidate_pass]
            self._candidate_pass += 1
            label = f'candidate_pass={self._candidate_pass} lambda={scale}'
        x.progress(label + ' begin', 0, tick, force=True)
        for batch in x:
            h, spikes = super().hidden(batch, w, u)
            features.append(h)
            done += len(batch)
            tick = x.progress(label, done, tick, force=done == len(x))
            del spikes, batch
        return torch.cat(features), None

    def upgradeSCN(self, x, target):
        if not isinstance(x, ChunkedInput):
            return super().upgradeSCN(x, target)
        # Like native upgradeSCN: run hidden once, fit globally, reuse spikes.
        # Reuse exact binary spikes under the same shared cache memory budget.
        results, features, shapes = [], [], []
        done, tick = 0, time.monotonic()
        x.progress('readout features begin', 0, tick, force=True)
        with BinaryCache(x.budget, x.runtime.output if x.runtime else None) as cache:
            for index, batch in enumerate(x):
                h, spikes = super().hidden(batch, self.W, self.U)
                features.append(h)
                shapes.append(tuple(spikes.shape))
                context = x.runtime.preprocessing() if x.runtime else nullcontext()
                with context:
                    cache.put(index, spikes)
                done += len(batch)
                tick = x.progress('readout features', done, tick, force=done == len(x))
                del spikes, batch
            self.computeBeta(torch.cat(features), target)
            done, tick = 0, time.monotonic()
            x.progress('readout loss begin', 0, tick, force=True)
            for index, shape in enumerate(shapes):
                context = x.runtime.preprocessing() if x.runtime else nullcontext()
                with context:
                    spikes = cache.get(index, shape, self.device)
                drive = torch.einsum('abc,cd->abd', spikes, self.Beta.float())
                filt = drive.new_zeros((shape[0], target.shape[1]))
                output, history = filt.clone(), []
                for t in range(x.shape[1]):
                    filt, output = self.alpha * filt + drive[:, t], self.beta * output + filt
                    history.append(output)
                results.append(self.softmax(torch.stack(history, 1).sum(1)))
                done += shape[0]
                tick = x.progress('readout loss', done, tick, force=done == len(x))
                del spikes, drive, history
        result = torch.cat(results)
        self.COST = (target - result).square().mean(0).mean()
        return result, self.COST
