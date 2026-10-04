import unittest
import torch
from pp3.flops import FlopCounter


class NativeAccountingTests(unittest.TestCase):
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
