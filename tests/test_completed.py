import csv
import json
from pathlib import Path
import tempfile
import unittest
from pp3.completed import identity, completed_index, completed_status
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
