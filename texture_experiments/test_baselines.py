"""Small correctness checks independent of real training data and GPU hardware."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from energy import AuxiliaryMemory
from models import Classifier
from train import evaluate


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name('defaults.json').read_text())
        torch.manual_seed(57)

    def test_gradient_reaches_each_hidden_layer(self):
        for method in ('rnn', 'rsnn', 'snnsurrogate', 'ann'):
            with self.subTest(method=method):
                model = Classifier(method, self.config)
                logits = model(torch.randn(3, 7, 9))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
                params = list(model.hidden.parameters())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                self.assertGreater(sum(p.grad.abs().sum().item() for p in params), 0)
                if method == 'rsnn':
                    self.assertGreater(model.recurrent.weight.grad.abs().sum().item(), 0)

    def test_state_resets_and_ann_is_order_invariant(self):
        x = torch.randn(3, 7, 9)
        for method in ('rnn', 'rsnn', 'snnsurrogate', 'ann'):
            model = Classifier(method, self.config).eval()
            first = model(x)
            model(torch.randn_like(x))
            torch.testing.assert_close(first, model(x))
            if method == 'ann':
                torch.testing.assert_close(first, model(x.flip(1)))

    def test_metrics_weight_unequal_batches(self):
        model = Classifier('ann', self.config)
        x, y = torch.randn(5, 7, 9), torch.tensor([0, 1, 2, 3, 4])
        result = evaluate(model, DataLoader(TensorDataset(x, y), batch_size=3), torch.device('cpu'))
        with torch.no_grad():
            logits = model(x)
            expected = torch.nn.functional.mse_loss(logits.softmax(-1),
                        torch.nn.functional.one_hot(y, 12).float()).item()
        self.assertAlmostEqual(result['mse'], expected, places=7)
        self.assertAlmostEqual(result['acc'], (logits.argmax(-1) == y).float().mean().item())

    def test_auxiliary_memory_subtracts_only_resident_parameters(self):
        with patch('torch.cuda.synchronize'), \
             patch('torch.cuda.reset_peak_memory_stats'), \
             patch('torch.cuda.max_memory_allocated', return_value=4096):
            with AuxiliaryMemory(torch.device('cuda:0'), 1024) as meter:
                pass
        result = meter.result()
        self.assertEqual(meter.peak_device_allocated_bytes, 4096)
        self.assertEqual(meter.parameter_device_bytes, 1024)
        self.assertEqual(result['auxiliary_mem_bytes'], 3072)
        self.assertNotIn('peak_device_allocated_bytes', result)
        self.assertNotIn('parameter_device_bytes', result)

    def test_cpu_auxiliary_memory_is_explicitly_unavailable(self):
        with AuxiliaryMemory(torch.device('cpu'), 0) as meter:
            pass
        result = meter.result()
        self.assertIsNone(result['auxiliary_mem_bytes'])
        self.assertIn('CPU', result['auxiliary_mem_reason'])


if __name__ == '__main__':
    unittest.main()
