"""Failure-path output must expose unknown counts, not a plausible total."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from constants import make_config
from pp3.worker import execute


class WorkerTests(unittest.TestCase):
    def test_unknown_epoch_saved_in_csv(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            c=make_config('spsn','spike')
            c.update(smoke=True,device='cpu',output=str(root/'out'))
            c['data']['root']=str(root)
            c['model'].update(hidden_size=8,layers=1,epochs=1,batch_size=3)
            for split in ('train','test'):
                np.save(root/c['data'][split+'_x'],np.ones((4,400,64),dtype='float32'))
                np.save(root/c['data'][split+'_y'],np.zeros(4,dtype='int64'))
            def unknown_batch(config,model,optimizers,x,y,train,counter=None):
                out=x[:,0,:20].sin()
                return out,out.sum()
            with patch('pp3.worker.batch',side_effect=unknown_batch):
                with self.assertRaisesRegex(RuntimeError,'Uncovered train'):
                    execute(c)
            with (root/'out'/'metrics.csv').open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['train_complete'],'False')
            self.assertEqual(rows[0]['train_flops'],'')
            self.assertGreater(int(rows[0]['train_known_flops']),0)
            report=json.loads((root/'out'/'epoch_0001_train_operations.json').read_text())
            self.assertTrue(any('sin' in r['operator'] for r in report['unknown']))


if __name__=='__main__':
    unittest.main()
