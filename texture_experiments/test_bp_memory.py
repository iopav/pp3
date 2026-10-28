"""Exact storage examples, lifetime checks, and gradient/update parity."""
import copy
import gc
import unittest
import torch
from bp_memory import SavedTensorMemory
from models import Classifier


class SaveAliases(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, parameter):
        ctx.save_for_backward(x, x[:, :1], x.view(-1), parameter, parameter.t())
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        _ = ctx.saved_tensors
        return grad * 2, None


class BPMemoryTests(unittest.TestCase):
    def test_alias_dedup_parameter_exclusion_and_backward_release(self):
        x = torch.randn(3, 5, requires_grad=True)
        p = torch.nn.Parameter(torch.randn(5, 5))
        with SavedTensorMemory([p], 'cpu') as meter:
            loss = SaveAliases.apply(x, p).sum()
            self.assertEqual(meter.current_bytes, 3 * 5 * 4)
            loss.backward(retain_graph=True)
            self.assertEqual(meter.current_bytes, 60)
            loss.backward()
        self.assertEqual(meter.peak_bytes, 60)
        self.assertEqual(meter.current_bytes, 0)
        torch.testing.assert_close(x.grad, torch.full_like(x, 4))

    def test_graph_discard_does_not_leak_or_accumulate_old_batches(self):
        with SavedTensorMemory([], 'cpu') as meter:
            for _ in range(3):
                x = torch.randn(10, requires_grad=True)
                result = x.exp()  # Saves a graph output, which must be detached in pack.
                self.assertEqual(meter.current_bytes, 40)
                del result
                gc.collect()
                self.assertEqual(meter.current_bytes, 0)
        self.assertEqual(meter.peak_bytes, 40)

    def test_inference_has_no_saved_backward_storage(self):
        with SavedTensorMemory([], 'cpu') as meter, torch.inference_mode():
            x = torch.randn(4, 8)
            _ = x.sigmoid() @ torch.randn(8, 2)
        self.assertEqual(meter.peak_bytes, 0)

    def check_model_parity(self, device):
        config = dict(hidden_size=4, rnn_nonlinearity='tanh', spike_beta=.9,
                      spike_threshold=1., surrogate_slope=25.)
        for method in ('ann', 'rnn', 'rsnn', 'snnsurrogate'):
            with self.subTest(method=method, device=device):
                torch.manual_seed(57)
                reference = Classifier(method, config, inputs=3, classes=2).to(device)
                measured = copy.deepcopy(reference)
                x = torch.randn(3, 5, 3, device=device)
                y = torch.tensor([0, 1, 0], device=device)
                a = reference(x)
                torch.nn.functional.cross_entropy(a, y).backward()
                with SavedTensorMemory(measured.parameters(), device) as meter:
                    b = measured(x)
                    torch.nn.functional.cross_entropy(b, y).backward()
                self.assertGreater(meter.peak_bytes, 0)
                self.assertEqual(meter.current_bytes, 0)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                for p, q in zip(reference.parameters(), measured.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                torch.optim.SGD(reference.parameters(), lr=.01).step()
                torch.optim.SGD(measured.parameters(), lr=.01).step()
                for p, q in zip(reference.parameters(), measured.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_all_baselines_cpu_parity(self):
        self.check_model_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_all_baselines_cuda_parity(self):
        self.check_model_parity('cuda:0')


if __name__ == '__main__':
    unittest.main()
