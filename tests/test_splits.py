import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from constants import make_config
from pp3.data import build_loaders
from pp3.splits import outer_split


class SplitTests(unittest.TestCase):
    def test_cached_indices_are_read_without_regenerating(self):
        with tempfile.TemporaryDirectory() as folder:
            config=make_config('spsn','spike');config['split_dir']=folder
            labels=np.repeat([0,1],10)
            first=outer_split(labels,config)
            config['method']='tp'
            with patch('pp3.splits._generate',side_effect=AssertionError('Must reuse indices')):
                second=outer_split(labels,config)
            self.assertTrue(np.array_equal(first[0],second[0]))
            self.assertTrue(np.array_equal(first[1],second[1]))
            with self.assertRaisesRegex(ValueError,'differ from data/policy'):
                outer_split(labels+1,config)
    def test_only_gatezo_reserves_npy_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            labels=np.repeat([0,1],10)
            for split in ['train','test']:
                np.save(root/(split+'_x.npy'),np.zeros((20,4,2),dtype=np.float32))
                np.save(root/(split+'_y.npy'),labels)
            seen=[]
            for method in ['spsn','tp','gatezo','ffsnn','espp','efficienttrain','bdnn']:
                config=make_config(method,'spike')
                config['data'].update(root=folder,steps=4,inputs=2,classes=2,
                    train_x='train_x.npy',train_y='train_y.npy',test_x='test_x.npy',test_y='test_y.npy')
                config['output']=str(root/method)
                train,test,_=build_loaders(config)
                self.assertEqual(len(test.dataset),20)
                if method!='gatezo':
                    self.assertEqual(len(train.dataset),20)
                    np.testing.assert_array_equal(train.dataset.origin_indices, np.arange(20))
                    self.assertFalse(hasattr(train.dataset, 'validation_dataset'))
                    self.assertFalse((root/method/'split.json').exists())
                else:
                    self.assertEqual(len(train.dataset),16)
                    self.assertFalse((root/method/'split.json').exists())
                    manifest=json.loads((root/'_splits/spike/seed_57.json').read_text())
                    seen.append(manifest['split_sha256'])
                    self.assertFalse(set(manifest['train_indices'])&set(manifest['validation_indices']))
                    if method=='gatezo':
                        self.assertEqual(len(train.dataset.validation_dataset),4)
                train.dataset.close();test.dataset.close()
                if method=='gatezo': train.dataset.validation_dataset.close()
            self.assertEqual(len(set(seen)),1)

    def test_shd_same_as_original_gatezo(self):
        from gate_zo_bdnn.train_shd_bdnn_rsnn_blocks import speaker_disjoint_split
        speakers=np.repeat(['a','b','c','d'],10)
        config=make_config('spsn','shd');config.pop('output',None)
        labels=np.tile([0,1],20)
        for seed in range(57,62):
            config['seed']=seed
            actual=outer_split(labels,config,speakers)
            expected=speaker_disjoint_split(speakers,2,seed)
            self.assertTrue(np.array_equal(actual[0],expected[0]))
            self.assertTrue(np.array_equal(actual[1],expected[1]))
            self.assertFalse(set(speakers[actual[0]])&set(speakers[actual[1]]))

    def test_shd_non_gatezo_keeps_all_original_samples(self):
        from pp3.data import EventFrames
        from types import SimpleNamespace
        class Source:
            labels = np.tile([0, 1], 10)
            speakers = np.repeat(['a', 'b', 'c', 'd'], 5)
            def __len__(self):
                return len(self.labels)
        with patch('gate_zo_bdnn.train_shd_bdnn_rsnn_pure.SHDEvents', return_value=Source()), \
             patch('gate_zo_bdnn.train_shd_bdnn_rsnn_pure.SHDBinner',
                   return_value=SimpleNamespace(steps=2000, channels=700)), \
             patch('pp3.splits.outer_split', side_effect=AssertionError('Unused holdout')):
            for method in ['spsn', 'tp', 'ffsnn', 'espp', 'efficienttrain', 'bdnn']:
                config = make_config(method, 'shd')
                config['data']['jitter_copies'] = 0
                dataset = EventFrames(config['data'], True, 57, config=config)
                np.testing.assert_array_equal(dataset.origin_indices, np.arange(20))
                self.assertEqual(len(dataset), 20)

    def test_too_small_class_is_explicit_error(self):
        config=make_config('tp','texture');config.pop('output',None)
        with self.assertRaisesRegex(ValueError,'at least two'):
            outer_split(np.array([0,1,1]),config)
