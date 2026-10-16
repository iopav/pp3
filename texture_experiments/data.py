"""Accept unified Texture files or the original bundle's two six-class splits."""
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

DEFAULT_DATA_ROOT = Path('D:/workspace/Avalanche/avalanche-master/dataset/texture'
                         if os.name == 'nt' else '/home/yizhil/pp3/data/texture-mini')


class Texture(Dataset):
    def __init__(self, x_path, y_path, steps=None, limit=None):
        self.x = np.load(x_path, mmap_mode='r', allow_pickle=False)
        self.y = np.load(y_path, mmap_mode='r', allow_pickle=False)
        if self.x.ndim != 3 or self.x.shape[1:] != (3775, 9):
            raise ValueError(f'{x_path}: expected (N,3775,9), got {self.x.shape}')
        if (self.y.shape != (len(self.x),) or not len(self.y)
                or not np.issubdtype(self.y.dtype, np.integer)
                or self.y.min() < 0 or self.y.max() > 11):
            raise ValueError(f'{y_path}: expected integer labels 0..11 matching X')
        self.steps = steps or 3775
        self.count = min(limit, len(self.y)) if limit else len(self.y)

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        x = np.array(self.x[i, :self.steps], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite Texture input at sample {i}')
        return torch.from_numpy(x), int(self.y[i])


def datasets(root, smoke=False):
    root = Path(root)
    kwargs = dict(steps=16, limit=8) if smoke else {}
    if (root / 'X_train_texture.npy').exists():
        return tuple(Texture(root / f'X_{part}_texture.npy',
                             root / f'Y_{part}_texture.npy', **kwargs)
                     for part in ('train', 'test'))
    return tuple(ConcatDataset([
        Texture(root / f'{age}6classes/X_{part}_6cla_{age}.npy',
                root / f'{age}6classes/Y_{part}_6cla_{age}.npy', **kwargs)
        for age in ('old', 'new')]) for part in ('train', 'test'))
