import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / 'BDNNversion1-main/measure_cpu_gpu.py'
spec = importlib.util.spec_from_file_location('power_meter', PATH)
meter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = meter
spec.loader.exec_module(meter)

class PowerMathTests(unittest.TestCase):
    def test_trapezoid_and_idle_subtraction(self):
        sensor = meter.Sensor('CPU', 'power', 'CPU socket', 'test', lambda: 0)
        idle = meter.summarize(sensor, [(0,10),(1,10),(2,10)], 1)
        run = meter.summarize(sensor, [(0,20),(1,40),(2,20)], 1)
        self.assertEqual(run['energy_j'],60)
        self.assertEqual(run['average_power_w'],30)
        self.assertEqual(run['energy_j']-idle['average_power_w']*run['covered_seconds'],40)

    def test_counter_wrap_and_reset(self):
        sensor = meter.Sensor('CPU', 'energy', 'CPU socket', 'test', lambda: 0, 100)
        self.assertEqual(meter.summarize(sensor, [(0,95),(1,5)], 1)['energy_j'],10)
        sensor.wrap = None
        self.assertIsNone(meter.summarize(sensor, [(0,95),(1,5)], 1)['energy_j'])

    def test_missing_sample_and_long_gap_not_integrated(self):
        sensor = meter.Sensor('GPU', 'power', 'GPU device', 'test', lambda: 0)
        result = meter.summarize(sensor, [(0,20),(1,None),(2,20),(10,20),(11,20)], 1)
        self.assertEqual(result['covered_seconds'],1)
        self.assertEqual(result['energy_j'],20)
        self.assertEqual(result['excluded_intervals'],3)
        self.assertIsNone(meter.summarize(sensor,[(0,20)],1)['average_power_w'])

if __name__ == '__main__':
    unittest.main()
