"""Artificial HDF5 events for entrypoint smoke checks; NOT the SHD dataset."""
from pathlib import Path
import h5py
import numpy as np

root = Path(__file__).resolve().parent/'fixtures'/'artificial_shd_20class'
root.mkdir(parents=True, exist_ok=True)
for split in ('train', 'test'):
    examples = [(speaker, label, repeat) for speaker in range(4 if split == 'train' else 1)
                for label in range(20) for repeat in range(2)]
    with h5py.File(root/f'shd_{split}.h5', 'w') as handle:
        group = handle.create_group('spikes')
        times = group.create_dataset('times', (len(examples),), dtype=h5py.vlen_dtype(np.dtype('float32')))
        units = group.create_dataset('units', (len(examples),), dtype=h5py.vlen_dtype(np.dtype('int64')))
        for index, (speaker, label, repeat) in enumerate(examples):
            times[index] = np.linspace(0, .999, 64, dtype=np.float32)
            units[index] = (np.arange(64)+label*31+repeat*3) % 700
        handle['labels'] = np.asarray([row[1] for row in examples], dtype=np.int64)
        handle['extra/speaker'] = np.asarray([row[0] for row in examples], dtype=np.int64)
(root/'ARTIFICIAL_DATA.txt').write_text('Synthetic events for interface tests only. Not real SHD.\n')
print(root)
