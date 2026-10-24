"""Regress the standalone Texture power-integration meter."""
from contextlib import contextmanager, ExitStack
import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


MODULES = ('texture_experiments.energy',)


@contextmanager
def environment(module, powers=(100000, 100000), shutdown_delay=0):
    clock = SimpleNamespace(now=10.)
    nv = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(return_value='allocated-device'),
        nvmlDeviceGetTotalEnergyConsumption=Mock(return_value=9000),
        nvmlDeviceGetPowerUsage=Mock(side_effect=powers))

    class Sampler:
        ident = 123

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self):
            clock.now += shutdown_delay

    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, pynvml=nv))
        stack.enter_context(patch.object(module.time, 'perf_counter', lambda: clock.now))
        stack.enter_context(patch.object(module.threading, 'Thread', Sampler))
        stack.enter_context(patch('torch.cuda.synchronize'))
        stack.enter_context(patch('torch.cuda.get_device_properties',
                                 return_value=SimpleNamespace(uuid='GPU-assigned')))
        yield clock, nv


class EnergyWindowTests(unittest.TestCase):
    def test_prefers_instantaneous_power_over_averaged_api(self):
        for name in MODULES:
            module = importlib.import_module(name)
            with self.subTest(module=name), environment(module) as (clock, nv):
                nv.NVML_FI_DEV_POWER_INSTANT = 186
                nv.NVML_SUCCESS = 0
                nv.NVML_VALUE_TYPE_UNSIGNED_INT = 1
                nv.nvmlDeviceGetFieldValues = Mock(return_value=[SimpleNamespace(
                    nvmlReturn=0, valueType=1, value=SimpleNamespace(uiVal=50000))])
                with module.Energy(torch.device('cuda:0')) as meter:
                    clock.now += .02
                self.assertAlmostEqual(meter.joules, 1.)
                self.assertIn('NVML_FI_DEV_POWER_INSTANT', meter.result()['energy_reason'])
                nv.nvmlDeviceGetPowerUsage.assert_not_called()

    def test_unsupported_instantaneous_field_falls_back_explicitly(self):
        for name in MODULES:
            module = importlib.import_module(name)
            with self.subTest(module=name), environment(module) as (clock, nv):
                nv.NVML_FI_DEV_POWER_INSTANT = 186
                nv.NVML_SUCCESS = 0
                nv.nvmlDeviceGetFieldValues = Mock(return_value=[SimpleNamespace(nvmlReturn=3)])
                with module.Energy(torch.device('cuda:0')) as meter:
                    clock.now += .02
                self.assertAlmostEqual(meter.joules, 2.)
                self.assertIn('may be time-averaged', meter.result()['energy_reason'])

    def test_short_epoch_with_stale_counter_has_bounded_estimate(self):
        for name in MODULES:
            module = importlib.import_module(name)
            with self.subTest(module=name), environment(module) as (clock, nv):
                with module.Energy(torch.device('cuda:0'), .1) as meter:
                    clock.now += .02
                result = meter.result()
                self.assertAlmostEqual(result['gpu_energy_j'], 2.)
                self.assertAlmostEqual(result['gpu_energy_j'] / result['seconds'], 100.)
                self.assertEqual(result['energy_method'], 'power_integral_estimate')
                self.assertIn('not independently resolved', result['energy_reason'])
                nv.nvmlDeviceGetTotalEnergyConsumption.assert_not_called()
                nv.nvmlDeviceGetHandleByUUID.assert_called_once_with('GPU-assigned')
                nv.nvmlShutdown.assert_called_once()

    def test_varying_power_integral_excludes_shutdown_and_late_samples(self):
        for name in MODULES:
            module = importlib.import_module(name)
            with self.subTest(module=name), environment(
                    module, (100000, 200000, 100000), shutdown_delay=5) as (clock, nv):
                with module.Energy(torch.device('cuda:0')) as meter:
                    clock.now += .01
                    meter.sample()
                    # Simulate a background sample completing after the end boundary.
                    meter.samples.append((11., 999999.))
                    clock.now += .01
                self.assertAlmostEqual(meter.seconds, .02)
                self.assertAlmostEqual(meter.joules, 3.)

    def test_read_failures_remain_missing_instead_of_partial_or_zero(self):
        for name in MODULES:
            module = importlib.import_module(name)
            for powers in ((RuntimeError('sensor lost'), 100000),
                           (100000, RuntimeError('sensor lost')),
                           (100000, float('nan')), (100000, -1)):
                with self.subTest(module=name, powers=powers), environment(
                        module, powers) as (clock, nv):
                    with module.Energy(torch.device('cuda:0')) as meter:
                        clock.now += .02
                    self.assertIsNone(meter.result()['gpu_energy_j'])
                    self.assertTrue(meter.result()['energy_reason'])
                    nv.nvmlShutdown.assert_called_once()

    def test_multiple_epochs_do_not_absorb_evaluation_gaps(self):
        for name in MODULES:
            module = importlib.import_module(name)
            with self.subTest(module=name), environment(
                    module, (100000,) * 6) as (clock, nv):
                energies = []
                for _ in range(3):
                    with module.Energy(torch.device('cuda:0')) as meter:
                        clock.now += .02
                    energies.append(meter.joules)
                    clock.now += 1.  # Outside-window evaluation/reporting.
                self.assertAlmostEqual(sum(energies), 6.)
                self.assertTrue(all(e > 0 for e in energies))

    def test_cpu_and_invalid_interval(self):
        for name in MODULES:
            module = importlib.import_module(name)
            with self.subTest(module=name):
                with module.Energy(torch.device('cpu')) as meter:
                    pass
                self.assertIsNone(meter.result()['gpu_energy_j'])
                for value in (0, -1, float('nan'), float('inf')):
                    with self.assertRaises(ValueError):
                        module.Energy(torch.device('cpu'), value)


if __name__ == '__main__':
    unittest.main()
