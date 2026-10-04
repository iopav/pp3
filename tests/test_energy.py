import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from pp3.energy import Energy


class EnergyTests(unittest.TestCase):
    def test_power_fallback_uuid_and_integral(self):
        class Unsupported(Exception):
            pass
        handles=[]
        def missing_total(handle):
            raise Unsupported('no energy counter')
        def by_uuid(uuid):
            handles.append(uuid)
            return 'correct-card'
        nv=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,NVMLError=Unsupported,
            nvmlDeviceGetHandleByUUID=by_uuid,nvmlDeviceGetTotalEnergyConsumption=missing_total,
            nvmlDeviceGetPowerUsage=lambda handle:50000)
        with patch.dict(sys.modules,{'pynvml':nv}), \
             patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(uuid='GPU-test')), \
             patch('torch.cuda.synchronize'):
            with Energy(torch.device('cuda:0'),.005) as meter:
                time.sleep(.02)
            result=meter.result()
        self.assertEqual(handles,['GPU-test'])
        self.assertEqual(result['energy_method'],'power_integral_estimate')
        duration=meter.samples[-1][0]-meter.samples[0][0]
        self.assertAlmostEqual(result['gpu_energy_j'],50*duration)
        self.assertGreater(result['gpu_energy_j'],0)

    def test_cpu_blank_energy(self):
        with Energy(torch.device('cpu')) as meter:
            pass
        self.assertIsNone(meter.result()['gpu_energy_j'])
        self.assertIn('CPU',meter.result()['energy_reason'])


if __name__=='__main__':
    unittest.main()
