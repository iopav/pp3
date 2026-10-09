import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import torch
from pp3.flops import FlopCounter


class NativeAccountingTests(unittest.TestCase):
    def test_status_depends_only_on_detected_missing_rules(self):
        from pp3.native_runtime import Runtime
        for enabled, missing in ((True, False), (True, True), (False, False)):
            with self.subTest(enabled=enabled, missing=missing), tempfile.TemporaryDirectory() as directory:
                runtime = Runtime.__new__(Runtime)
                runtime.output = Path(directory)
                runtime.config = dict(method='tp', dataset='spike', seed=57, count_flops=enabled)
                runtime.device = torch.device('cpu')
                runtime.rows, runtime.results, runtime.unknown_records = [], {}, []
                runtime.phase, runtime.epoch, runtime.data_wait = 'train', 1, 0.
                runtime.energy = MagicMock()
                runtime.energy.result.return_value = {}
                runtime.counter = FlopCounter(enabled=enabled)
                value = torch.ones(3)
                runtime.counter.__enter__()
                if missing:
                    torch.lgamma(value)
                else:
                    value + value
                runtime.end()
                runtime.finish({})
                status = json.loads((runtime.output/'status.json').read_text())
                self.assertEqual(status['status'], 'flops_incomplete' if missing else 'success')
                self.assertNotIn('flops_missing_calls', status)
                self.assertEqual(status['flops_complete'], (not missing) if enabled else None)

    def test_missing_calls_persist_per_shape_and_phase(self):
        from pp3.native_runtime import Runtime
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime.__new__(Runtime)
            runtime.output = Path(directory)
            runtime.phase, runtime.epoch = 'train', 2
            runtime.data_wait, runtime.complete, runtime.results = 0., True, {}
            runtime.unknown_records = []
            runtime.energy = MagicMock()
            runtime.energy.result.return_value = {}
            runtime.counter = FlopCounter()
            small, large = torch.ones(3), torch.ones(5)
            runtime.counter.__enter__()
            torch.lgamma(small)
            torch.lgamma(small)
            torch.lgamma(large)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                runtime.end()
            report = json.loads((Path(directory)/'flops_unknown.json').read_text())
            records = report['records']
            self.assertEqual(report['calls_by_operator'], {'aten.lgamma.default': 3})
            self.assertEqual(sorted(r['calls'] for r in records), [1, 2])
            self.assertEqual({tuple(r['inputs'][0]['shape']) for r in records}, {(3,), (5,)})
            self.assertTrue(all(r['epoch'] == 2 and r['phase'] == 'train' for r in records))
            self.assertEqual(log.getvalue(), '')
            self.assertFalse(runtime.complete)

    def test_normal_sampling_estimate_and_rng_transparency(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for size in (0, 1, 3, 16, 17):
                for kind in ('normal_', 'randn', 'randn_like'):
                    with self.subTest(device=device, size=size, kind=kind):
                        states = []
                        for enabled in (False, True):
                            value = torch.empty(size, device=device)
                            torch.manual_seed(57)
                            counter = FlopCounter(enabled=enabled)
                            with counter:
                                if kind == 'normal_':
                                    result = value.normal_(2., 3.)
                                elif kind == 'randn':
                                    result = torch.randn(size, device=device)
                                else:
                                    result = torch.randn_like(value)
                            rng = (torch.cuda.get_rng_state() if device == 'cuda'
                                   else torch.random.get_rng_state())
                            states.append((result, rng))
                        self.assertTrue(torch.equal(states[0][0], states[1][0]))
                        self.assertTrue(torch.equal(states[0][1], states[1][1]))
                        summary = counter.summary()
                        self.assertTrue(summary['complete'], summary['unknown'])
                        self.assertEqual(summary['flops'], 9*((size+1)//2)+2*size)
                        self.assertEqual(summary['other_flops_estimated'], summary['flops'])

    def test_complex_normal_remains_uncovered(self):
        value = torch.empty(3, dtype=torch.complex64)
        with FlopCounter() as counter:
            value.normal_()
        self.assertFalse(counter.summary()['complete'])

    def test_unknown_dispatched_operator_still_fails_coverage(self):
        value = torch.ones(3)
        counter = FlopCounter()
        with counter:
            torch.lgamma(value)
        result = counter.summary()
        self.assertFalse(result['complete'])
        self.assertIsNone(result['flops'])
        self.assertTrue(any('lgamma' in item['operator'] for item in result['unknown']))

    def test_matrix_vector_and_fused_scaling(self):
        matrix, vector, bias = torch.ones(2,3), torch.ones(3), torch.ones(2)
        for operation, expected in (
                (lambda: matrix @ vector, 12),
                (lambda: torch.addmv(bias, matrix, vector), 14),
                (lambda: torch.addmv(bias, matrix, vector, alpha=2, beta=3), 18),
                (lambda: torch.addmv(bias, matrix, vector, beta=0), 12)):
            counter = FlopCounter()
            with counter:
                operation()
            self.assertTrue(counter.summary()['complete'])
            self.assertEqual(counter.summary()['flops'], expected)

    def test_excluding_scalar_bookkeeping_does_not_hide_missing_major_work(self):
        counter = FlopCounter()
        counter.mark_excluded('python_scalar_bookkeeping', 'loop arithmetic')
        self.assertTrue(counter.summary()['complete'])
        self.assertEqual(len(counter.summary()['excluded']), 1)
        counter.mark_unknown('numpy::pinv', 'uncovered major solver')
        self.assertFalse(counter.summary()['complete'])
        self.assertIsNone(counter.summary()['flops'])

    def test_batchnorm_reference_estimate(self):
        x=torch.ones(2,3,2,2); w=torch.ones(3); b=torch.zeros(3)
        mean=torch.zeros(3); var=torch.ones(3)
        counter=FlopCounter()
        with counter:
            output,saved_mean,saved_inv=torch.ops.aten.native_batch_norm.default(x,w,b,mean,var,True,.1,1e-5)
        self.assertEqual(counter.summary()['flops'],8*24+9*3+5)
        grad=torch.ones_like(x)
        counter=FlopCounter()
        with counter:
            torch.ops.aten.native_batch_norm_backward.default(grad,x,w,mean,var,saved_mean,saved_inv,True,1e-5,[True,True,True])
        self.assertEqual(counter.summary()['flops'],9*24+4*3+2)
        self.assertGreater(counter.summary()['other_flops_estimated'],0)

    def test_lu_solve_estimate(self):
        matrix=torch.eye(3);rhs=torch.ones(3,2)
        counter=FlopCounter()
        with counter:
            torch.linalg.solve(matrix,rhs)
        self.assertEqual(counter.summary()['flops'],18+36)
    def test_foreach_inplace_void_return_and_per_tensor_scale(self):
        destination = [torch.zeros(2), torch.zeros(3)]
        left = [torch.ones(2), torch.ones(3)]
        right = [torch.ones(2), torch.ones(3)]
        scalar = torch.tensor(1.)
        counter = FlopCounter()
        with counter:
            torch._foreach_add_(destination, scalar)
            torch._foreach_addcdiv_(destination, left, right, [1., .5])
            torch._foreach_lerp_(destination, left, .5)
            torch._foreach_sqrt(destination)
        self.assertEqual(counter.summary()['flops'], 5+(2*2+3*3)+3*5+5)
        self.assertTrue(counter.summary()['complete'])

    def test_silu_equivalent(self):
        x = torch.ones(4, requires_grad=True)
        counter = FlopCounter()
        with counter:
            torch.nn.functional.silu(x).sum().backward()
        # SiLU 2N, summation N-1, product-rule backward 6N.
        self.assertEqual(counter.summary()['flops'], 2*4+3+6*4)

    def test_convolution_forward_and_requested_backward(self):
        x = torch.ones(1, 1, 4, 4)
        weight = torch.ones(1, 1, 2, 2)
        bias = torch.zeros(1)
        counter = FlopCounter()
        with counter:
            output = torch.ops.aten.convolution.default(x, weight, bias, [1,1], [0,0], [1,1], False, [0,0], 1)
        self.assertEqual(counter.summary()['flops'], 81)
        grad = torch.ones_like(output)
        counter = FlopCounter()
        with counter:
            torch.ops.aten.convolution_backward.default(grad, x, weight, [1], [1,1], [0,0], [1,1], False, [0,0], 1, [True,True,True])
        self.assertEqual(counter.summary()['flops'], 152)

    def test_scatter_and_standard_deviation(self):
        x = torch.zeros(3)
        source = torch.ones(4)
        index = torch.tensor([0, 1, 1, 2])
        counter = FlopCounter()
        with counter:
            x.scatter_add_(0, index, source)
            torch.std(source.reshape(2,2), dim=1, correction=0)
        self.assertEqual(counter.summary()['flops'], 4+18)

    def test_disabled_counter_does_not_change_random_draws(self):
        states = []
        for enabled in (True, False):
            torch.manual_seed(57)
            with FlopCounter(enabled=enabled):
                result = torch.rand(4)*2-1
            states.append((result, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(states[0][0], states[1][0]))
        self.assertTrue(torch.equal(states[0][1], states[1][1]))


if __name__ == '__main__':
    unittest.main()
