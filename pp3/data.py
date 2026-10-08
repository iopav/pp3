"""Top-level CPU data preparation. No downloads, disk conversion or caches."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def balanced_indices(labels, limit):
    """Round-robin classes for smoke tests only; full runs keep source order."""
    labels = np.asarray(labels)
    pools = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    chosen, position = [], 0
    while len(chosen) < min(limit, len(labels)):
        for pool in pools:
            if position < len(pool):
                chosen.append(int(pool[position]))
                if len(chosen) == limit:
                    break
        position += 1
    return np.asarray(chosen, dtype=np.int64)


class ArrayDataset(Dataset):
    def __init__(self, root, x, y, spec, limit=None, steps=None, indices=None):
        self.x_path, self.y_path = str(Path(root)/x), str(Path(root)/y)
        self.x = self.y = None
        self.open()
        if self.x.ndim != 3 or tuple(self.x.shape[1:]) != (spec['steps'], spec['inputs']):
            raise ValueError(f'{self.x_path}: expected (N,{spec["steps"]},{spec["inputs"]}), got {self.x.shape}')
        if self.y.shape != (len(self.x),) or not np.issubdtype(self.y.dtype, np.integer):
            raise ValueError('Labels must be an integer vector matching X')
        if not len(self.y) or self.y.min() < 0 or self.y.max() >= spec['classes']:
            raise ValueError('Empty dataset or labels outside configured classes')
        self.indices=np.arange(len(self.y)) if indices is None else np.asarray(indices,dtype=np.int64)
        self.count = min(len(self.indices), limit) if limit else len(self.indices)
        if limit:
            # A tiny smoke split still needs distinct classes for contrastive samplers.
            self.indices=self.indices[balanced_indices(self.y[self.indices], self.count)]
        self.targets=np.asarray(self.y[self.indices],dtype=np.int64)
        self.origin_indices=self.indices.copy()
        self.is_augmented=np.zeros(self.count,dtype=bool)
        self.time_indices = np.linspace(0, spec['steps']-1, steps, dtype=int) if steps else slice(None)

    def open(self):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode='r', allow_pickle=False)
            self.y = np.load(self.y_path, mmap_mode='r', allow_pickle=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['x'] = state['y'] = None
        return state

    def close(self):
        for name in ('x','y'):
            array=getattr(self,name,None)
            if array is not None and getattr(array,'_mmap',None) is not None:
                array._mmap.close()
            setattr(self,name,None)

    def __del__(self):
        self.close()

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        self.open()
        index=int(self.indices[i])
        a = np.array(self.x[index, self.time_indices], dtype=np.float32, copy=True)
        if not np.isfinite(a).all():
            raise ValueError(f'Non-finite input in {self.x_path} sample {i}')
        return torch.from_numpy(a), int(self.y[index])


class EventFrames(Dataset):
    def __init__(self, spec, train, seed, limit=None, steps=None, config=None):
        from gate_zo_bdnn.train_shd_bdnn_rsnn_pure import SHDEvents, SHDBinner
        from gate_zo_bdnn.train_shd_bdnn_rsnn_blocks import DeterministicJitterAugmentedSubset
        source = SHDEvents(Path(spec['root'])/('shd_train.h5' if train else 'shd_test.h5'))
        if not len(source) or source.labels.min()<0 or source.labels.max()>=spec['classes']:
            raise ValueError('Invalid SHD labels')
        indices = np.arange(len(source))
        if train and config and config['method'] == 'gatezo':
            from .splits import outer_split
            indices, _ = outer_split(source.labels, config, source.speakers)
        if limit:
            indices = indices[balanced_indices(source.labels[indices], limit)]
        from torch.utils.data import Subset
        if train and spec['jitter_copies'] and (spec['jitter_std'] or spec['time_scale_min']!=1 or spec['time_scale_max']!=1):
            self.source = DeterministicJitterAugmentedSubset(source, indices,
                spec['jitter_std'], spec['jitter_copies'], seed+65537,
                source_channels=700, time_scale_min=spec['time_scale_min'], time_scale_max=spec['time_scale_max'])
        else:
            self.source = Subset(source, indices.tolist())
        self.binner = SHDBinner(spec['dt'],spec['max_time'],700,spec['channel_group_size'])
        if (self.binner.steps,self.binner.channels)!=(spec['steps'],spec['inputs']):
            raise ValueError('SHD steps/inputs inconsistent with dt/max_time/channel grouping')
        self.time_indices = np.linspace(0,spec['steps']-1,steps,dtype=int) if steps else slice(None)
        self.targets=np.tile(source.labels[indices],len(self.source)//len(indices))
        self.origin_indices=np.tile(indices,len(self.source)//len(indices))
        self.is_augmented=np.arange(len(self.source))>=len(indices)
        self.speakers=(None if source.speakers is None else np.tile(source.speakers[indices],len(self.source)//len(indices)))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        x,y = self.binner([self.source[i]])
        return x[0,self.time_indices], int(y[0])


class FloatImageDataset(Dataset):
    """Input-format bridge for the original EfficientTrain ResNet entry.

    Time and channel axes form a 2-D plane, replicated over RGB, then resized.
    This is not a spiking model; float32 negative values are preserved.
    """
    def __init__(self, source, size):
        self.source, self.size = source, int(size)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, label = self.source[index]
        x = torch.nn.functional.interpolate(x[None, None], size=(self.size, self.size),
                                           mode='bilinear', align_corners=False)[0]
        return x.expand(3, -1, -1).contiguous(), label

def build_loaders(config):
    start = time.perf_counter()
    spec, smoke = config['data'], config.get('smoke',False)
    limit, steps = (config.get('smoke_samples',4),config.get('smoke_steps',16)) if smoke else (None,None)
    sets=[]
    for train in (True,False):
        if config['dataset']=='shd':
            ds = EventFrames(spec,train,config['seed'],limit,steps,config=config)
        else:
            prefix='train' if train else 'test'
            indices = None
            if train and config['method'] == 'gatezo':
                from .splits import outer_split
                labels=np.load(Path(spec['root'])/spec['train_y'],allow_pickle=False)
                indices, validation_indices=outer_split(labels,config)
            ds=ArrayDataset(spec['root'],spec[prefix+'_x'],spec[prefix+'_y'],spec,limit,steps,indices)
            if train and config['method']=='gatezo':
                ds.validation_dataset=ArrayDataset(spec['root'],spec['train_x'],spec['train_y'],
                    spec,limit,steps,validation_indices)
        sets.append(ds)
    generator=torch.Generator().manual_seed(config['seed'])
    loaders=[DataLoader(ds,batch_size=config['model']['batch_size'],shuffle=(i==0),
               num_workers=config['workers'],generator=generator if i==0 else None,
               drop_last=False, pin_memory=False) for i,ds in enumerate(sets)]
    return *loaders, time.perf_counter()-start
