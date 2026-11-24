"""Bounded shared RAM budget for exact binary data; at most one spill file/store."""
import tempfile
import numpy as np
import torch


class CacheBudget:
    def __init__(self, limit=8 * 1024**3):
        self.limit, self.used = int(limit), 0
        if self.limit < 0:
            raise ValueError('Cache budget must be nonnegative')


class BinaryCache:
    def __init__(self, budget, directory=None):
        self.budget, self.directory = budget, directory
        self.records, self.file = {}, None
        self.ram_bytes = 0

    def __contains__(self, key):
        return key in self.records

    def put(self, key, tensor):
        if key in self.records:
            raise ValueError('Cache key already exists')
        array = tensor.detach().cpu().numpy()
        if not np.all((array == 0) | (array == 1)):
            raise ValueError('Binary cache requires exact 0/1 values')
        packed = np.packbits(array.astype(np.uint8).reshape(-1))
        if self.budget.used + packed.nbytes <= self.budget.limit:
            self.records[key] = packed
            self.budget.used += packed.nbytes
            self.ram_bytes += packed.nbytes
        else:
            if self.file is None:
                self.file = tempfile.TemporaryFile(prefix='.bdnn_cache_', dir=self.directory)
            self.file.seek(0, 2)
            offset = self.file.tell()
            self.file.write(packed.tobytes())
            self.records[key] = (offset, packed.nbytes)

    def get(self, key, shape, device):
        packed = self.records[key]
        if isinstance(packed, tuple):
            offset, length = packed
            self.file.seek(offset)
            data = self.file.read(length)
            if len(data) != length:
                raise OSError('Truncated binary cache')
            packed = np.frombuffer(data, dtype=np.uint8)
        array = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape)
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)

    def close(self):
        self.records.clear()
        self.budget.used -= self.ram_bytes
        self.ram_bytes = 0
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
