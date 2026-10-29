import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch
from pp3.native_runtime import Runtime


class BDNNGPUAccountingTests(unittest.TestCase):
    def measure(self, method, readings):
        nv = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByUUID=Mock(return_value='allocated-gpu'),
            nvmlDeviceGetTotalEnergyConsumption=Mock(side_effect=readings))
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {'pynvml':nv}), \
                patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(uuid='GPU-allocated')), \
                patch('torch.cuda.synchronize'), patch('pp3.energy.time.perf_counter', side_effect=[10.,12.,12.,15.]):
            r = Runtime.__new__(Runtime)
            r.config = dict(method=method, device='cuda:0', power_sample_seconds=.1, count_flops=False)
            r.device = torch.device('cpu' if method == 'bdnn' else 'cuda:0')
            r.output = Path(root)
            r.counter = r.energy = None
            r.results, r.rows, r.unknown_records = {}, [], []
            for phase in ('train','test'):
                r.begin(phase,1)
                r.end(.5)
            r.epoch_row(1)
            with (Path(root)/'metrics.csv').open(encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(r.device.type, 'cpu' if method == 'bdnn' else 'cuda')
            nv.nvmlDeviceGetHandleByUUID.assert_called_with('GPU-allocated')
            return row

    def test_all_methods_use_same_energy_formula_and_csv_fields(self):
        reference = self.measure('spsn', [1000,5000,5000,11000])
        self.assertEqual(float(reference['train_gpu_energy_j']),4.)
        self.assertEqual(float(reference['test_gpu_energy_j']),6.)
        self.assertEqual(float(reference['epoch_gpu_energy_j']),10.)
        self.assertEqual(float(reference['epoch_seconds']),5.)
        for method in ('bdnn','tp','ffsnn','espp','gatezo','efficienttrain'):
            with self.subTest(method=method):
                self.assertEqual(self.measure(method,[1000,5000,5000,11000]), reference)

    def test_missing_sensor_has_same_behavior_as_other_methods(self):
        for readings in ([RuntimeError('not supported')]*2, [1000,900,1000,900]):
            bdnn = self.measure('bdnn',readings)
            other = self.measure('spsn',readings)
            self.assertEqual(bdnn,other)
            self.assertEqual(bdnn['epoch_gpu_energy_j'],'')

if __name__ == '__main__':
    unittest.main()
