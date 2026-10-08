import csv
import json
from pathlib import Path
import tempfile
import unittest

from pp3.reporting import export_summary


class ReportingTests(unittest.TestCase):
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
            self.assertEqual(rows[0]['source_unit'],'growth_round')
            self.assertEqual(rows[-1]['test_acc'],'0.5')
