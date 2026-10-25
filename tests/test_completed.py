import csv
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from pp3.completed import identity, completed_index, completed_status, source_digest
from summarize_experiments import summarize


def fixture(out, **changes):
    out.mkdir(parents=True)
    config=dict(method='spsn',dataset='spike',seed=57,smoke=True,source_sha256='v1',
                execution_device='cpu',model=dict(epochs=1),output=str(out))
    config.update(changes); config['experiment_id']=identity(config)
    (out/'launch.json').write_text(json.dumps(config))
    (out/'model.pt').write_bytes(b'fake')
    (out/'status.json').write_text(json.dumps(dict(status='flops_incomplete',
        training_completed=True,model_bytes=4,epochs=1)))
    (out/'metrics.csv').write_text('epoch,epoch_flops,epoch_known_flops,epoch_flops_complete\n1,,12,False\n')
    (out/'epoch_0001_train_operations.json').write_text('{"complete": false}')
    return config


class CompletedTests(unittest.TestCase):
    def test_cleaned_v2_and_v3_ignore_coverage_but_validate_training_artifacts(self):
        for version in (2, 3):
            for coverage in (True, False, None):
                with self.subTest(version=version, coverage=coverage), tempfile.TemporaryDirectory() as folder:
                    out=Path(folder)/'task'; config=fixture(out)
                    metric='epoch_known_flops' if version==2 else 'epoch_flops'
                    csv_path=out/'metrics.csv'
                    csv_path.write_text(f'epoch,{metric},final_model_bytes\n1,12,4\n')
                    original=csv_path.read_bytes()
                    status=dict(artifact_version=version,experiment_id=config['experiment_id'],
                        status='flops_incomplete',training_completed=True,model_bytes=4,epochs=1)
                    if coverage is not None:
                        status['flops_complete']=coverage
                    (out/'status.json').write_text(json.dumps(status))
                    self.assertIsNotNone(completed_status(out))
                    self.assertIn(config['experiment_id'],completed_index(out))
                    self.assertEqual(csv_path.read_bytes(),original)
                    for change in ({'training_completed':False},{'status':'failed'},
                                   {'epochs':2},{'model_bytes':5}):
                        (out/'status.json').write_text(json.dumps(dict(status,**change)))
                        self.assertIsNone(completed_status(out))

    def test_legacy_does_not_require_operator_coverage_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'task'; fixture(out)
            (out/'epoch_0001_train_operations.json').unlink()
            self.assertIsNotNone(completed_status(out))

    def test_controller_skips_cleaned_v2_without_launch_and_returns_success(self):
        import run_experiments
        with tempfile.TemporaryDirectory() as folder:
            argv=['run_experiments.py','--cpu','--methods','spsn','--datasets','spike',
                  '--seeds','57','--output',folder]
            capture=io.StringIO()
            with patch('sys.argv',argv+['--dry-run']), contextlib.redirect_stdout(capture):
                self.assertEqual(run_experiments.main(),0)
            config=json.loads(capture.getvalue())['tasks'][0]
            out=Path(config['output']); fixture(out,**config)
            (out/'metrics.csv').write_text('epoch,epoch_known_flops,final_model_bytes\n1,12,4\n')
            (out/'status.json').write_text(json.dumps(dict(artifact_version=2,
                experiment_id=config['experiment_id'],status='flops_incomplete',
                training_completed=True,flops_complete=False,model_bytes=4,epochs=1)))
            capture=io.StringIO()
            with patch('sys.argv',argv), patch.object(run_experiments.subprocess,'Popen') as launch, \
                 contextlib.redirect_stdout(capture):
                self.assertEqual(run_experiments.main(),0)
            launch.assert_not_called()
            self.assertIn(f'跳过 [skip completed] spsn/spike/57: {out}',capture.getvalue())
            self.assertNotIn('epoch_flops_complete',(out/'metrics.csv').read_text())

    def test_old_resnet_baseline_is_not_accepted_as_efficienttrain(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'old_et'
            fixture(out, method='efficienttrain')
            status=json.loads((out/'status.json').read_text())
            status['method']='efficienttrain'
            (out/'status.json').write_text(json.dumps(status))
            self.assertIsNone(completed_status(out))
            self.assertFalse(completed_index(Path(folder)))

    def test_compact_artifacts_skip_and_preserve_one_time_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); out=root/'task'
            config=fixture(out)
            (out/'launch.json').unlink()
            (out/'epoch_0001_train_operations.json').unlink()
            (out/'metrics.csv').write_text('epoch,epoch_known_flops,epoch_flops_complete,final_model_bytes\n1,12,False,4\n')
            status=dict(artifact_version=2, experiment_id=config['experiment_id'],
                        method='spsn',dataset='spike',seed=57,status='flops_incomplete',
                        training_completed=True,model_bytes=4,epochs=1,
                        one_time_metrics=[dict(phase='final_refit',known_flops=7)])
            (out/'status.json').write_text(json.dumps(status))
            self.assertIn(config['experiment_id'], completed_index(root))
            destination=summarize([root],root/'summary.csv')
            with destination.open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([r['flops'] for r in rows], ['12','7'])
            metadata=json.loads(destination.with_suffix('.metadata.json').read_text(encoding='utf-8'))
            self.assertEqual([r['record_type'] for r in metadata['rows']], ['epoch','final_refit'])
            self.assertTrue(all(r['experiment_id']==config['experiment_id'] for r in metadata['rows']))
            (out/'model.pt').write_bytes(b'broken')
            self.assertFalse(completed_index(root))

    def test_shared_numpy_accounting_changes_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'method').mkdir()
            (root/'method/main.py').write_text('# entry')
            (root/'pp3').mkdir()
            for name in ('data','splits','native_runtime','flops','numpy_linalg','energy'):
                (root/'pp3'/f'{name}.py').write_text('# initial')
            before = source_digest(root, root/'method/main.py')
            (root/'pp3/numpy_linalg.py').write_text('# revised accounting')
            self.assertNotEqual(before, source_digest(root, root/'method/main.py'))

    def test_complete_incomplete_flops_skip_but_changed_config_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=fixture(root/'run1'/'task')
            index=completed_index(root)
            self.assertIn(config['experiment_id'],index)
            for change in [dict(seed=58),dict(smoke=False),dict(source_sha256='v2'),
                           dict(execution_device='cuda'),dict(model=dict(epochs=2))]:
                other=dict(config,**change)
                self.assertNotIn(identity(other),index)
            (root/'run1/task/model.pt').write_bytes(b'bad')
            self.assertFalse(completed_index(root))

    def test_discovery_without_parent_manifest_and_summary_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fixture(root/'run1/task');fixture(root/'run2/task')
            broken=root/'run3/task';fixture(broken,seed=58)
            (broken/'status.json').unlink()
            self.assertIsNone(completed_status(broken))
            result=summarize([root],root/'summary.csv')
            with result.open(encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['flops'],'12')
            selection=json.loads(result.with_suffix('.selection.json').read_text())
            self.assertEqual(len(selection['excluded']),1)
