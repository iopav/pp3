import csv
import json
from pathlib import Path
import tempfile
import unittest
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
            self.assertEqual([(r['record_type'],r['flops']) for r in rows], [('epoch','12'),('final_refit','7')])
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
