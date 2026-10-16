import csv
import json
from pathlib import Path
import tempfile
import unittest

from pp3.reporting import export_summary


class ReportingTests(unittest.TestCase):
    def test_same_labels_different_configs_preserve_identity_and_missing_energy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            tasks=[]
            for identity in ('config_a','config_b'):
                out=root/identity
                out.mkdir()
                tasks.append(dict(method='tp',dataset='spike',seed=57,
                                  experiment_id=identity,output=str(out)))
                (out/'status.json').write_text(json.dumps(dict(artifact_version=2,
                    status='success',model_bytes=16,one_time_metrics=[
                        dict(phase='prepare',gpu_energy_j=None,known_flops=5,
                             energy_reason='CPU run: GPU energy unavailable')])),encoding='utf-8')
                (out/'metrics.csv').write_text(
                    'epoch,epoch_gpu_energy_j,epoch_flops,train_accuracy,test_accuracy\n'
                    '1,,12,0.75,0.5\n',encoding='utf-8')
            destination=export_summary(root,tasks=tasks)
            with destination.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            metadata=json.loads(destination.with_suffix('.metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows),4)
            self.assertEqual([row['energy_j'] for row in rows],['']*4)
            self.assertEqual([row['flops'] for row in rows],['12','5','12','5'])
            self.assertEqual([row['epoch'] for row in rows],['1','','1',''])
            self.assertEqual([row['experiment_id'] for row in metadata['rows']],
                             ['config_a','config_a','config_b','config_b'])
            self.assertEqual(metadata['rows'][1]['energy_note'],'CPU run: GPU energy unavailable')

    def test_missing_flops_and_final_cost_not_repeated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root/'task'
            task.mkdir()
            (root/'tasks.json').write_text(json.dumps([dict(method='gatezo',dataset='spike',
                seed=57,output=str(task))]), encoding='utf-8')
            (task/'status.json').write_text(json.dumps(dict(status='flops_incomplete',
                model_bytes=1048576)), encoding='utf-8')
            with (task/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['epoch','unit','epoch_gpu_energy_j',
                    'epoch_flops','epoch_known_flops','epoch_flops_complete'])
                writer.writeheader()
                for epoch in [1,2]:
                    writer.writerow(dict(epoch=epoch,unit='growth_round',epoch_gpu_energy_j=4,
                        epoch_flops='',epoch_known_flops=100,epoch_flops_complete=False))
            (task/'epoch_0000_final_refit_and_test_operations.json').write_text(json.dumps(
                dict(gpu_energy_j=7,flops=None,known_flops=20,complete=False,accuracy=.5)),encoding='utf-8')
            destination=export_summary(root)
            with destination.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(sum(float(r['energy_j']) for r in rows),15)
            self.assertEqual([r['flops'] for r in rows],['100','100','20'])
            self.assertNotIn('known_flops',rows[0])
            self.assertNotIn('flops_complete',rows[0])
            self.assertEqual(rows[0]['mem_mib'],'1.0')
            for field in ('experiment_id','record_type','source_unit','energy_note','output'):
                self.assertNotIn(field,rows[0])
            metadata=json.loads(destination.with_suffix('.metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['rows'][0]['source_unit'],'growth_round')
            self.assertEqual([r['csv_data_row'] for r in metadata['rows']],[1,2,3])
            self.assertEqual([r['record_type'] for r in metadata['rows']],
                             ['epoch','epoch','final_refit_and_test'])
            self.assertEqual(rows[-1]['test_acc'],'0.5')
