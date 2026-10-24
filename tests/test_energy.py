import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import torch
from pp3.energy import Energy


class EnergyTests(unittest.TestCase):
    def test_counter_difference_uuid_and_no_power_fallback(self):
        class Unsupported(Exception):
            pass
        handles=[]
        def by_uuid(uuid):
            handles.append(uuid)
            return 'correct-card'
        nv=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,NVMLError=Unsupported,
            nvmlDeviceGetHandleByUUID=by_uuid,
            nvmlDeviceGetTotalEnergyConsumption=Mock(side_effect=[1000, 15500]),
            nvmlDeviceGetPowerUsage=Mock())
        with patch.dict(sys.modules,{'pynvml':nv}), \
             patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(uuid='GPU-test')), \
             patch('torch.cuda.synchronize'):
            with Energy(torch.device('cuda:0'),.005) as meter:
                time.sleep(.02)
            result=meter.result()
        self.assertEqual(handles,['GPU-test'])
        self.assertEqual(result['energy_method'],'nvml_total')
        self.assertEqual(result['gpu_energy_j'],14.5)
        self.assertEqual(nv.nvmlDeviceGetTotalEnergyConsumption.call_count,2)
        nv.nvmlDeviceGetPowerUsage.assert_not_called()

    def test_zero_unsupported_and_reset_counter(self):
        for readings, expected in (([1000,1000], 0.),
                ([RuntimeError('not supported')], None), ([1000,900], None)):
            nv=SimpleNamespace(nvmlInit=Mock(),nvmlShutdown=Mock(),
                nvmlDeviceGetHandleByUUID=Mock(return_value='assigned-card'),
                nvmlDeviceGetTotalEnergyConsumption=Mock(side_effect=readings),
                nvmlDeviceGetPowerUsage=Mock())
            with self.subTest(readings=readings), patch.dict(sys.modules,{'pynvml':nv}), \
                 patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(uuid='GPU-test')), \
                 patch('torch.cuda.synchronize'):
                with Energy(torch.device('cuda:0')) as meter:
                    pass
                self.assertEqual(meter.result()['gpu_energy_j'],expected)
                nv.nvmlDeviceGetPowerUsage.assert_not_called()
                nv.nvmlShutdown.assert_called_once()

    def test_cpu_blank_energy(self):
        with Energy(torch.device('cpu')) as meter:
            pass
        self.assertIsNone(meter.result()['gpu_energy_j'])
        self.assertIn('CPU',meter.result()['energy_reason'])


if __name__=='__main__':
    unittest.main()
