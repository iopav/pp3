"""Successful-run reuse must not conceal changed or incomplete experiments."""
import json
from pathlib import Path
import tempfile
import unittest
from completion import skip_completed


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'run'
        self.config = dict(method='ann', epochs=2, dataset='texture-mini', seed=57,
                           batch_size=128, data_root='/data/texture-mini',
                           device_name='GPU A', torch_version='old')

    def write_run(self):
        self.path.mkdir()
        self.write('config.json', self.config)
        self.write('status.json', dict(status='success', epochs=2))
        (self.path / 'model.pt').write_bytes(b'checkpoint placeholder')
        (self.path / 'metrics.csv').write_text('epoch\n1\n2\n')

    def write(self, name, value):
        (self.path / name).write_text(json.dumps(value), encoding='utf-8')

    def test_new_run_and_matching_completed_run(self):
        self.assertFalse(skip_completed(self.path, self.config))
        self.write_run()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.path.iterdir()}
        self.assertTrue(skip_completed(self.path, self.config))
        self.assertTrue(skip_completed(self.path, dict(self.config, device_name='GPU B', torch_version='new')))
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.path.iterdir()})

    def test_changed_configuration_is_not_skipped(self):
        self.write_run()
        for key, value in [('epochs', 3), ('batch_size', 64), ('dataset', 'shd'),
                           ('data_root', '/another/path'), ('seed', 42)]:
            with self.subTest(key=key), self.assertRaisesRegex(FileExistsError, 'configuration differs'):
                skip_completed(self.path, dict(self.config, **{key: value}))

    def test_failed_missing_or_corrupt_status_is_not_skipped(self):
        self.write_run()
        for value in [dict(status='failed'), dict(status='running'), [], dict(status='success', epochs=1)]:
            self.write('status.json', value)
            with self.assertRaises(FileExistsError):
                skip_completed(self.path, self.config)
        (self.path / 'status.json').write_text('{')
        with self.assertRaises(FileExistsError):
            skip_completed(self.path, self.config)
        (self.path / 'status.json').unlink()
        with self.assertRaises(FileExistsError):
            skip_completed(self.path, self.config)

    def test_missing_artifact_is_not_skipped(self):
        self.write_run()
        (self.path / 'model.pt').unlink()
        with self.assertRaisesRegex(FileExistsError, 'model.pt'):
            skip_completed(self.path, self.config)

    def test_gatezo_requires_final_outputs_and_target(self):
        self.config = dict(method='gatezo', max_hidden=49, native_args={'lambdas': (1, 2)})
        self.write_run()
        self.write('status.json', dict(status='success', hidden_neurons=49))
        with self.assertRaises(FileExistsError):
            skip_completed(self.path, self.config)
        self.write('summary.json', {})
        self.write('final_metrics.json', {})
        self.assertTrue(skip_completed(self.path, self.config))
        self.write('status.json', dict(status='success', hidden_neurons=48))
        with self.assertRaises(FileExistsError):
            skip_completed(self.path, self.config)


if __name__ == '__main__':
    unittest.main()
