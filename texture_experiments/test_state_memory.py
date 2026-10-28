import copy
import unittest
import torch
from state_memory import make_optimizer, optimizer_state_bytes


class StateMemoryTests(unittest.TestCase):
    def test_momentum_storage_dedup_and_restored_next_update(self):
        model = torch.nn.Linear(3, 2)
        config = dict(optimizer='SGD', lr=.01, momentum=.9)
        opt = make_optimizer(model.parameters(), config)
        self.assertEqual(optimizer_state_bytes(opt), 0)
        x = torch.ones(4, 3)
        model(x).square().mean().backward()
        opt.step()
        expected = sum(p.numel()*p.element_size() for p in model.parameters())
        self.assertEqual(optimizer_state_bytes(opt), expected)
        clone = copy.deepcopy(model)
        restored = make_optimizer(clone.parameters(), config)
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        for net, optimizer in ((model, opt), (clone, restored)):
            optimizer.zero_grad(set_to_none=True)
            net(x).square().mean().backward()
            optimizer.step()
        for p, q in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        state = next(iter(opt.state.values()))
        state['alias_for_test'] = state['momentum_buffer'].view(-1)
        self.assertEqual(optimizer_state_bytes(opt), expected)

    def test_adam_counts_actual_moments_and_step_tensors(self):
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        opt = make_optimizer([p], dict(optimizer='Adam', lr=.01))
        p.sum().backward()
        opt.step()
        state = opt.state[p]
        expected = sum(v.numel()*v.element_size() for v in state.values())
        self.assertEqual(optimizer_state_bytes(opt), expected)


if __name__ == '__main__':
    unittest.main()
