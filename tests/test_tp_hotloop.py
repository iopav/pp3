"""Check TP hot-loop optimizations against the original update sequence."""
import ast
import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'traces_propagation-main'))
from models.tp_mlp import TP_MLP


def train_function(legacy=False):
    source = (ROOT / 'traces_propagation-main/utils.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'train')
    if legacy:
        class RestoreLosses(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == 'not custom_grad or t == time_steps - 1':
                    return node.body
                return self.generic_visit(node)
        node = RestoreLosses().visit(node)
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<train>', 'exec'), namespace)
    return namespace['train']


class LegacyTP(TP_MLP):
    def _spike_summer(self, i, student, teacher):
        self.cumulative_spikes[i] += student.sum().item()
        self.cumulative_spikes_target[i] += teacher.sum().item()
        self.total_spike_neurons[i] += student.numel()
        self.total_spike_neurons_target[i] += teacher.numel()


class CountingCE(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args):
        self.calls += 1
        return super().forward(*args)


class HotLoopTests(unittest.TestCase):
    def check_device(self, device):
        torch.manual_seed(57)
        params = dict(l_leak_t=.38, surrogate_type='1', surrogate_scale=10.,
                      train_s=True, l_vth=.7, l_leak_m=.11, l_rec=True,
                      l_rst_type='Soft', l_out_leak_m=1., norm_type='weight')
        old = LegacyTP([5, 4], 3, 2, 3, params).to(device)
        new = TP_MLP([5, 4], 3, 2, 3, params).to(device)
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        batches = [(torch.rand(b, 7, 3), torch.arange(b) % 2) for b in (3, 2)]
        results, histories, losses = [], [], []
        for model, legacy in ((old, True), (new, False)):
            optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
            history = []
            optimizer.register_step_post_hook(lambda opt, args, kwargs:
                history.append([p.detach().clone() for p in model.parameters()]))
            ce = CountingCE()
            results.append(train_function(legacy)(model, batches, device, 2,
                {'all': optimizer}, ce, True, 1, 0, 1, False, 'TP', 'SHD'))
            histories.append(history)
            losses.append(ce.calls)
        self.assertEqual(losses, [28, 4])
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(histories[0]), 14)
        for a, b in zip(histories[0], histories[1]):
            for pa, pb in zip(a, b):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        self.assertEqual(old.spike_count_summary(), new.spike_count_summary())
        for value in new.cumulative_spikes + new.cumulative_spikes_target:
            self.assertEqual(value.device.type, torch.device(device).type)
            self.assertEqual(value.dtype, torch.int64)
            self.assertIsNone(value.grad_fn)
            self.assertFalse(value.requires_grad)

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
