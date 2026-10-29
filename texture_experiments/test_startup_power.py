import contextlib
import io
import os
import unittest
from unittest.mock import Mock, patch
from startup_power import MARKER, report_startup_power


class StartupPowerTests(unittest.TestCase):
    def test_cpu_prints_once_and_is_not_zero(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stream):
            result = report_startup_power('cpu')
            report_startup_power('cpu')
        self.assertIsNone(result['power_w'])
        self.assertEqual(stream.getvalue().count('[startup_power]'), 1)

    def test_mean_and_no_repeated_probe(self):
        nv = Mock()
        nv.nvmlDeviceGetPowerUsage.side_effect = [10000, 30000]
        with patch.dict(os.environ, {}, clear=True), patch.dict('sys.modules', pynvml=nv), \
             patch('torch.cuda.get_device_properties', return_value=Mock(uuid='GPU-test')), \
             patch('startup_power.time.perf_counter', side_effect=[0, 0, .1, .1, .2, .2]), \
             patch('startup_power.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            result = report_startup_power('cuda:0', seconds=.2)
            report_startup_power('cuda:0')
        self.assertEqual(result['power_w'], 20)
        self.assertEqual(result['samples'], 2)
        nv.nvmlShutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
