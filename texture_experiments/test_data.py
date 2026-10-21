"""Data contract, SHD binning, and non-Texture model dimensions."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import datasets, labels, Sequences, DEFAULT_DATA_ROOT
from models import Classifier
from train import evaluate


class DataTests(unittest.TestCase):
    def test_real_texture_preserved(self):
        path = DEFAULT_DATA_ROOT / 'texture_spikesT450_T77.npz'
        if not path.exists():
            self.skipTest('Local Texture-mini unavailable')
        train, test = datasets(DEFAULT_DATA_ROOT)
        self.assertEqual((len(train), len(test), train.steps, train.inputs, train.classes),
                         (960, 240, 77, 9, 12))
        with np.load(path) as source:
            for part, ds in [('train', train), ('test', test)]:
                np.testing.assert_array_equal(ds.y, source[f'Y_{part}'].argmax(1))
                for i in range(len(ds)):
                    np.testing.assert_array_equal(ds[i][0].numpy(), source[f'X_{part}'][i])

    def test_reject_bad_labels_and_shapes(self):
        for y in (np.zeros((2, 12)), np.ones((2, 12)), np.array([12]), np.array([1.5])):
            with self.assertRaises(ValueError):
                labels(y, 12)
        with self.assertRaises(ValueError):
            Sequences(np.zeros((2, 4, 3)), np.array([0]), 12)
        with tempfile.TemporaryDirectory() as root:
            np.savez(Path(root) / 'texture_spikesT450_T77.npz',
                     X_train=np.zeros((2, 4, 3)), Y_train=np.array([0, 1]),
                     X_test=np.zeros((2, 5, 3)), Y_test=np.array([0, 1]))
            with self.assertRaises(ValueError):
                datasets(root)

    def test_shd_bins_and_dynamic_models(self):
        import h5py
        with tempfile.TemporaryDirectory() as root:
            for part in ('train', 'test'):
                with h5py.File(Path(root) / f'shd_{part}.h5', 'w') as f:
                    f['labels'] = np.array([0, 19])
                    t = f.create_dataset('spikes/times', (2,), dtype=h5py.vlen_dtype(np.dtype('float64')))
                    u = f.create_dataset('spikes/units', (2,), dtype=h5py.vlen_dtype(np.dtype('int64')))
                    for i in range(2):
                        t[i] = np.array([0, 0.001, 0.005, 0.019, 0.02])
                        u[i] = np.array([0, 0, 1, 699, 0])
            train, _ = datasets(root, dataset='shd', shd_max_time=0.02, shd_channel_group_size=2)
            x, y = train[1]
            self.assertEqual((tuple(x.shape), y, x.sum().item()), ((4, 350), 19, 3))
            self.assertEqual((x[0, 0], x[1, 0], x[3, 349]), (1, 1, 1))
            config = json.loads(Path(__file__).with_name('defaults.json').read_text())
            for method in ('ann', 'rnn', 'rsnn', 'snnsurrogate'):
                model = Classifier(method, config, train.inputs, train.classes)
                logits = model(x[None])
                self.assertEqual(tuple(logits.shape), (1, 20))
                torch.nn.functional.cross_entropy(logits, torch.tensor([19])).backward()
                result = evaluate(model, DataLoader(train, batch_size=2), torch.device('cpu'))
                with torch.no_grad():
                    expected = torch.nn.functional.mse_loss(model(torch.stack([train[i][0] for i in range(2)])).softmax(-1),
                        torch.nn.functional.one_hot(torch.tensor([0, 19]), 20).float()).item()
                self.assertAlmostEqual(result['mse'], expected, places=7)


if __name__ == '__main__':
    unittest.main()
