"""Counted FLOPs include FFT/other once, even with unsupported operators."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch
from pp3.native_runtime import Runtime
from pp3.completed import completed_status, completed_index
from summarize_experiments import summarize
from texture_experiments.metrics import compact_metrics


class MetricsSchemaTests(unittest.TestCase):
    def test_partial_coverage_fft_other_and_v3_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Runtime.__new__(Runtime)
            runtime.output = Path(folder)
            runtime.config = dict(method='spsn', dataset='texture', seed=57,
                count_flops=True, power_sample_seconds=.1, experiment_id='schema-v3')
            runtime.device = torch.device('cpu')
            runtime.counter = runtime.energy = None
            runtime.rows, runtime.results, runtime.unknown_records = [], {}, []
            runtime.data_wait = 0.
            runtime.begin('train', 1)
            torch.fft.rfft(torch.ones(2, 8))  # 120 FFT FLOPs, already in total.
            torch.linalg.solve(torch.eye(3), torch.ones(3, 2))  # 54 other FLOPs.
            torch.lgamma(torch.ones(3))  # Unknown: must not blank the CSV total.
            train = runtime.end()
            self.assertFalse(train['complete'])
            self.assertEqual(train['known_flops'], 174)
            runtime.begin('test', 1)
            torch.ones(3) + torch.ones(3)
            runtime.end()
            runtime.epoch_row(1, unit='growth_round')
            runtime.finish({'weight': torch.ones(2)})
            with (runtime.output/'metrics.csv').open(encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row['epoch_flops']), 177)
            self.assertEqual(float(row['train_flops']), 174)
            self.assertEqual(float(row['train_fft_flops_included']), 120)
            self.assertEqual(float(row['train_other_flops_estimated']), 54)
            self.assertFalse(any(key.endswith(('_complete', '_known_flops')) for key in row))
            self.assertNotIn('source_unit', row)
            status = completed_status(runtime.output)
            self.assertEqual(status['artifact_version'], 3)
            self.assertIn('schema-v3', completed_index(runtime.output))
            destination = summarize([folder], runtime.output/'summary.csv')
            with destination.open(encoding='utf-8-sig') as stream:
                summary_row = next(csv.DictReader(stream))
            self.assertEqual(float(summary_row['flops']), 177)
            self.assertNotIn('source_unit', json.loads(
                destination.with_suffix('.metadata.json').read_text())['rows'][0])

    def test_texture_included_header_without_diagnostics(self):
        result = compact_metrics(dict(epoch=1, hidden_state_mem_bytes=64,
            auxiliary_mem_bytes=128, auxiliary_mem_method='test',
            auxiliary_mem_reason='test', gpu_uuid='test'))
        self.assertEqual(result, {'epoch': 1, 'hidden_state_mem_bytes (included)': 64,
                                 'auxiliary_mem_bytes': 128})


if __name__ == '__main__':
    unittest.main()
