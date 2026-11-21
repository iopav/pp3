import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset
from bdnn_gpu.model import BDNNGPU
from bdnn_gpu.chunked import ChunkedInput, ChunkedBDNNGPU
from constants import make_config


class ChunkedTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required for memory measurement')
    def test_full_shd_time_axis_peak_memory(self):
        class Dataset:
            def __len__(self):
                return 24
            def __getitem__(self, index):
                return torch.ones(2000, 700), 0
        with torch.no_grad():
            settings = make_config('bdnn', 'shd')['model']
            dense = BDNNGPU(700, 2, settings, 'cuda')
            chunked = ChunkedBDNNGPU(700, 2, settings, 'cuda')
            w = torch.full((700, 4), .01, device='cuda')
            u = torch.zeros(4, 4, device='cuda')
            for model in (dense, chunked):
                model.W, model.U = w, u
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            x = torch.ones(24, 2000, 700, device='cuda')
            h, spikes = dense.getH(x)
            expected = h.cpu()
            torch.cuda.synchronize()
            dense_peak = torch.cuda.max_memory_allocated() - baseline
            del x, h, spikes
            torch.cuda.reset_peak_memory_stats()
            source = ChunkedInput(Dataset(), 'cuda', 8)
            h, _ = chunked.getH(source)
            torch.cuda.synchronize()
            chunk_peak = torch.cuda.max_memory_allocated() - baseline
            torch.testing.assert_close(expected, h.cpu())
            self.assertLess(chunk_peak, dense_peak * .8)
            print(f'2000-step input/hidden CUDA peaks: dense={dense_peak} chunked={chunk_peak}')

    def test_full_data_math_and_candidate_random_stream(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device), torch.no_grad():
                torch.manual_seed(57)
                settings = make_config('bdnn', 'shd')['model']
                settings.update(pool_size=4, growth_step=1, lambdas=[1, 3], r=[.99, .999])
                x = torch.rand(7, 19, 3) * 4
                target = torch.nn.functional.one_hot(torch.arange(7) % 2, 2).float().to(device)
                source = ChunkedInput(TensorDataset(x, torch.arange(7) % 2), device, 3)
                dense, chunked = BDNNGPU(3, 2, settings, device), ChunkedBDNNGPU(3, 2, settings, device)
                w = torch.rand(3, 4, device=device)
                u = torch.rand(4, 4, device=device) * .1
                for model in (dense, chunked):
                    model.W, model.U = w.clone(), u.clone()
                hd, _ = dense.getH(x.to(device))
                hc, spikes = chunked.getH(source)
                self.assertIsNone(spikes)
                torch.testing.assert_close(hd, hc)
                rd, cd = dense.upgradeSCN(x.to(device), target)
                rc, cc = chunked.upgradeSCN(source, target)
                torch.testing.assert_close(dense.Beta, chunked.Beta)
                torch.testing.assert_close(rd, rc)
                torch.testing.assert_close(cd, cc)
                torch.testing.assert_close(dense.getLabel(x.to(device)), chunked.getLabel(source))
                np.random.seed(57)
                a = dense.sc_Search(x.to(device), target)
                state_a = np.random.get_state()
                np.random.seed(57)
                b = chunked.sc_Search(source, target)
                state_b = np.random.get_state()
                np.testing.assert_array_equal(state_a[1], state_b[1])
                self.assertEqual(state_a[2:], state_b[2:])
                self.assertEqual(a is None, b is None)
                if a is not None:
                    for before, after in zip(a, b):
                        torch.testing.assert_close(before, after)
                for model in (dense, chunked):
                    model.W, model.U = w[:, :0], u[:0, :0]
                rd, cd = dense.upgradeSCN(x.to(device), target)
                rc, cc = chunked.upgradeSCN(source, target)
                torch.testing.assert_close(rd, rc)
                torch.testing.assert_close(cd, cc)

    def test_lazy_bounded_ordered_input(self):
        class Dataset:
            def __init__(self):
                self.visited = []
            def __len__(self):
                return 101
            def __getitem__(self, index):
                self.visited.append(index)
                return torch.full((5, 2), float(index)), 0
        data = Dataset()
        source = ChunkedInput(data, 'cpu', 8)
        self.assertEqual(data.visited, [0])
        data.visited.clear()
        sizes = []
        for batch in source:
            sizes.append(len(batch))
            self.assertLessEqual(len(batch), 8)
        self.assertEqual(sum(sizes), len(data))
        self.assertEqual(data.visited, list(range(len(data))))


if __name__ == '__main__':
    unittest.main()
