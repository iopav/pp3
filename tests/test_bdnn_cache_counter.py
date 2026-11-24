import tempfile
import unittest
from pathlib import Path
import torch
from bdnn_gpu.binary_cache import BinaryCache, CacheBudget
from bdnn_gpu.fast_counter import RepeatedOpCounter
from bdnn_gpu.model import BDNNGPU
from constants import make_config
from pp3.flops import FlopCounter


class CacheCounterTests(unittest.TestCase):
    def test_shared_budget_single_spill_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = CacheBudget(2)
            with BinaryCache(budget, folder) as a, BinaryCache(budget, folder) as b:
                x = torch.tensor([0., 1.] * 4)
                a.put(0, x)
                b.put(0, x)
                self.assertEqual(budget.used, 2)
                self.assertIsNone(a.file)
                self.assertIsNone(b.file)
                self.assertFalse(list(Path(folder).iterdir()))
                for i in range(1, 4):
                    b.put(i, x)
                self.assertIsNotNone(b.file)
                self.assertEqual(budget.used, 2)
                self.assertLessEqual(len(list(Path(folder).iterdir())), 1)
                for i in range(4):
                    torch.testing.assert_close(x, b.get(i, x.shape, 'cpu'), rtol=0, atol=0)
            self.assertEqual(budget.used, 0)
            self.assertFalse(list(Path(folder).iterdir()))

    def test_nonbinary_rejected_without_budget_leak(self):
        budget = CacheBudget(100)
        with BinaryCache(budget) as cache:
            with self.assertRaises(ValueError):
                cache.put(0, torch.tensor([.5]))
            self.assertEqual(budget.used, 0)

    def test_same_operator_flops_calls_and_unknowns(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            summaries = []
            outputs = []
            for counter_type in [FlopCounter, RepeatedOpCounter]:
                torch.manual_seed(57)
                model = BDNNGPU(3, 2, make_config('bdnn', 'shd')['model'], device)
                model.W = torch.rand(3, 4, device=device)
                model.U = torch.zeros(4, 4, device=device)
                model.counter = counter = counter_type()
                x = torch.rand(7, 21, 3, device=device)
                target = torch.nn.functional.one_hot(torch.arange(7, device=device) % 2, 2).float()
                with torch.no_grad(), counter:
                    with counter.section('training'):
                        output, loss = model.upgradeSCN(x, target)
                    with counter.section('diagnostic'):
                        x.add(x, alpha=2).sum(0)
                        x.add(x, alpha=3).sum(1)
                        torch.lgamma(x)  # Unknown must remain visible.
                outputs.append((output, loss))
                summary = counter.summary()
                totals = {}
                for row in summary['operations']:
                    key = (row['stage'], row['operator'])
                    f, calls = totals.get(key, (0, 0))
                    totals[key] = (f + row['flops'], calls + row['calls'])
                summaries.append((summary['known_flops'], totals,
                                  {row['operator'] for row in summary['unknown']}))
            self.assertEqual(summaries[0], summaries[1])
            for before, after in zip(*outputs):
                torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
