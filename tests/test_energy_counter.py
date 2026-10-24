"""Cumulative-counter backend must never substitute a power-based estimate."""
import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


class TotalEnergyCounterTests(unittest.TestCase):
    def test_short_epochs_merge_and_allocate_by_duration(self):
        for name in ('pp3.energy', 'texture_experiments.energy'):
            allocator = importlib.import_module(name).CounterWindowAllocator(1.)
            with self.subTest(module=name):
                self.assertEqual(allocator.add(1, (0., 1000), (.2, 1000)), {})
                result = allocator.add(2, (.2, 1000), (1., 11000))
                self.assertAlmostEqual(result[1]['gpu_energy_j'], 2.)
                self.assertAlmostEqual(result[2]['gpu_energy_j'], 8.)
                self.assertAlmostEqual(sum(v['gpu_energy_j'] for v in result.values()), 10.)

    def test_long_epoch_is_not_merged_with_previous_short_epochs(self):
        for name in ('pp3.energy', 'texture_experiments.energy'):
            allocator = importlib.import_module(name).CounterWindowAllocator(1.)
            with self.subTest(module=name):
                allocator.add(1, (0., 1000), (.2, 3000))
                result = allocator.add(2, (.2, 3000), (2.2, 23000))
                self.assertEqual(result[1]['gpu_energy_j'], 2.)
                self.assertEqual(result[2]['gpu_energy_j'], 20.)

    def test_pending_final_group_and_missing_counter_never_fake_zero(self):
        for name in ('pp3.energy', 'texture_experiments.energy'):
            for end in ((.2, 1000), (.2, None)):
                allocator = importlib.import_module(name).CounterWindowAllocator(1.)
                with self.subTest(module=name, end=end):
                    allocator.add(1, (0., 1000), end)
                    self.assertIsNone(allocator.flush()[1]['gpu_energy_j'])

    def test_stale_counter_and_millijoule_increment(self):
        for name in ('pp3.energy', 'texture_experiments.energy'):
            module = importlib.import_module(name)
            nv = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
                nvmlDeviceGetHandleByUUID=Mock(return_value='assigned-device'),
                nvmlDeviceGetTotalEnergyConsumption=Mock(side_effect=[1000, 1000, 15000]),
                nvmlDeviceGetPowerUsage=Mock())
            with self.subTest(module=name), patch.dict(sys.modules, pynvml=nv), \
                 patch('torch.cuda.synchronize') as synchronize, \
                 patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(uuid='GPU-test')):
                meter = module.TotalEnergyCounter(torch.device('cuda:0'))
                readings = [meter.read()[1] for _ in range(3)]
                meter.close()
                meter.close()
                self.assertEqual(readings, [1000, 1000, 15000])
                self.assertEqual((readings[-1] - readings[0]) / 1000, 14.)
                self.assertEqual(synchronize.call_count, 3)
                nv.nvmlDeviceGetHandleByUUID.assert_called_once_with('GPU-test')
                nv.nvmlDeviceGetPowerUsage.assert_not_called()
                nv.nvmlShutdown.assert_called_once()

    def test_reset_and_unsupported_counter_stay_unavailable(self):
        for name in ('pp3.energy', 'texture_experiments.energy'):
            module = importlib.import_module(name)
            for values in ([1000, 900], [1000, RuntimeError('not supported')]):
                nv = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
                    nvmlDeviceGetHandleByUUID=Mock(return_value='assigned-device'),
                    nvmlDeviceGetTotalEnergyConsumption=Mock(side_effect=values),
                    nvmlDeviceGetPowerUsage=Mock())
                with self.subTest(module=name, values=values), patch.dict(sys.modules, pynvml=nv), \
                     patch('torch.cuda.synchronize'), \
                     patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(uuid='GPU-test')):
                    meter = module.TotalEnergyCounter(torch.device('cuda:0'))
                    self.assertEqual(meter.read()[1], 1000)
                    self.assertIsNone(meter.read()[1])
                    self.assertIsNone(meter.read()[1])
                    self.assertEqual(nv.nvmlDeviceGetTotalEnergyConsumption.call_count, 2)
                    nv.nvmlDeviceGetPowerUsage.assert_not_called()
                    meter.close()


if __name__ == '__main__':
    unittest.main()
