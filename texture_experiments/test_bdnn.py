"""BDNN fixed-width protocol, failure evidence and final-test boundaries."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from bdnn import run_growth


class FakeBDNN:
    def __init__(self, fail_at=None):
        self.W, self.U, self.Beta = [], [], []
        self.L = 0
        self.fail_at = fail_at

    def sc_Search(self, x, targets):
        return np.ones((2, 1)), np.ones((1, 1)), int(self.L == self.fail_at)

    def addNodes(self, w, u):
        self.L += 1
        self.W = np.ones((2, self.L), dtype=np.float32)
        self.U = np.eye(self.L, dtype=np.float32)

    def getH(self, x):
        return np.ones((len(x), self.L)), None

    def computeBeta(self, h, targets):
        self.Beta = np.ones((self.L, 2), dtype=np.float64)

    def upgradeSCN(self, x, targets):
        self.computeBeta(*[self.getH(x)[0], targets])
        return self.getOutput(x), 0.5

    def getOutput(self, x):
        return np.ones((len(x), 2))


class BDNNTests(unittest.TestCase):
    def test_49_neurons_final_test_once_outside_window(self):
        counter = Mock()
        counter.read.side_effect = [(i, i * 1000) for i in range(22)]
        def test(model):
            self.assertEqual(counter.read.call_count, 22)
            self.assertEqual(model.L, 49)
            return .5
        callback = Mock(side_effect=test)
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            rows = run_growth(FakeBDNN(), np.ones((4, 3, 2)), np.eye(2)[[0, 1, 0, 1]],
                              49, counter, Path(root), callback)
        callback.assert_called_once()
        self.assertEqual(len(rows), 50)
        self.assertAlmostEqual(sum(r['energy_j'] for r in rows), 11)
        self.assertTrue(all(r['final_test_acc'] is None for r in rows[:-1]))
        self.assertEqual(rows[-1]['para_mem_bytes'], (2*49+49*49)*4 + 49*2*8)

    def test_failed_search_never_tests(self):
        counter = Mock()
        counter.read.side_effect = [(0, 0), (1, 1000)]
        callback = Mock()
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'candidate search failed'):
                run_growth(FakeBDNN(fail_at=1), np.ones((4, 3, 2)), np.eye(2)[[0, 1, 0, 1]],
                           49, counter, Path(root), callback)
            self.assertTrue((Path(root)/'metrics.csv').exists())
        callback.assert_not_called()


if __name__ == '__main__':
    unittest.main()
