import contextlib
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

from constants import make_config, SEEDS
from pp3.data import ArrayDataset, EventFrames, build_loaders
from pp3.flops import FlopCounter

ROOT=Path(__file__).resolve().parents[1]


def count(fn):
    counter=FlopCounter()
    with counter:
        result=fn()
    return result,counter.summary()


def create_h5(root):
    import h5py
    for split in ['train','test']:
        with h5py.File(Path(root)/f'shd_{split}.h5','w') as h:
            group=h.create_group('spikes')
            times=group.create_dataset('times',(4,),dtype=h5py.vlen_dtype(np.dtype('float32')))
            units=group.create_dataset('units',(4,),dtype=h5py.vlen_dtype(np.dtype('int64')))
            values=[[-.001,0.,0.,.0005,.9999,1.,1.05],[],[.01,.25,.75],[0.,.5]]
            channels=[[0,1,1,2,699,4,5],[],[10,20,30],[0,699]]
            for i in range(4):
                times[i]=np.asarray(values[i],dtype='float32'); units[i]=np.asarray(channels[i],dtype='int64')
            h['labels']=np.arange(4,dtype='int64')
            h['extra/speaker']=np.arange(4,dtype='int64')


class FlopsTests(unittest.TestCase):
    def test_manual_uniform_and_complex_magnitude(self):
        x=torch.empty(5)
        _,r=count(lambda: x.uniform_(-2,2))
        self.assertEqual(r['flops'],10)
        z=torch.tensor([1+2j,3+4j])
        _,r=count(lambda: z.abs())
        self.assertEqual(r['flops'],8)
        counter=FlopCounter()
        counter.add_manual('Adam',9,3)
        self.assertEqual(counter.summary()['flops'],27)

    def test_matmul_and_backward(self):
        a=torch.ones(2,3,requires_grad=True); b=torch.ones(3,4,requires_grad=True)
        _,s=count(lambda:(a@b).sum().backward())
        self.assertTrue(s['complete'],s['unknown'])
        # 48 forward + 48 each operand gradient + seven reduction additions.
        self.assertEqual(s['flops'],151)

    def test_broadcast_reduction_and_fused(self):
        a=torch.ones(2,3); b=torch.ones(3)
        _,s=count(lambda:(a+b).mean())
        self.assertEqual(s['flops'],12)
        _,s=count(lambda:torch.addcmul(a,a,a,value=2))
        self.assertEqual(s['flops'],18)

    def test_complex(self):
        a=torch.ones(5,dtype=torch.complex64)
        _,s=count(lambda:a*a+a)
        self.assertEqual(s['flops'],40)
        _,s=count(lambda:a*2.)
        self.assertEqual(s['flops'],10)

    def test_fft(self):
        x=torch.ones(2,8)
        _,s=count(lambda:torch.fft.rfft(x))
        self.assertTrue(s['complete'])
        self.assertEqual(s['fft_flops_estimated'],120)
        z=torch.fft.rfft(x)
        _,s=count(lambda:torch.fft.irfft(z,n=8))
        self.assertEqual(s['fft_flops_estimated'],136) # inverse normalization: 16 divides

    def test_unknown_not_zero(self):
        _,s=count(lambda:torch.sin(torch.ones(3)))
        self.assertFalse(s['complete']); self.assertIsNone(s['flops'])
        self.assertTrue(any('sin' in r['operator'] for r in s['unknown']))

    def test_dropout_not_blanket_zero(self):
        # Fused/native dropout must remain unknown unless its multiply has a formula.
        _,s=count(lambda:torch.ops.aten.native_dropout(torch.ones(9),.5,True))
        self.assertFalse(s['complete'])

    def test_metadata_zero(self):
        _,s=count(lambda:torch.ones(2,3).reshape(3,2).clone())
        self.assertTrue(s['complete']); self.assertEqual(s['flops'],0)

    def test_softmax(self):
        x=torch.ones(2,3)
        _,s=count(lambda:torch.softmax(x,dim=1))
        self.assertEqual(s['flops'],22)
        _,s=count(lambda:torch.log_softmax(x,dim=1))
        self.assertEqual(s['flops'],24)


class DataTests(unittest.TestCase):
    def test_npy_shapes_tail_negative(self):
        with tempfile.TemporaryDirectory() as root:
            x=np.arange(4*6*2,dtype='float64').reshape(4,6,2)-10
            np.save(Path(root)/'x.npy',x); np.save(Path(root)/'y.npy',np.arange(4))
            ds=ArrayDataset(root,'x.npy','y.npy',dict(steps=6,inputs=2,classes=4))
            self.assertEqual(ds[0][0].dtype,torch.float32)
            self.assertEqual(float(ds[0][0][0,0]),-10)
            self.assertEqual([len(y) for _,y in torch.utils.data.DataLoader(ds,batch_size=3)],[3,1])
            with self.assertRaises(ValueError):
                ArrayDataset(root,'x.npy','y.npy',dict(steps=5,inputs=2,classes=4))
            ds.close()

    def test_gatezo_exact_and_no_cache(self):
        from gate_zo_bdnn.train_shd_bdnn_rsnn_pure import SHDEvents,SHDBinner
        from gate_zo_bdnn.train_shd_bdnn_rsnn_blocks import DeterministicJitterAugmentedSubset
        with tempfile.TemporaryDirectory() as root:
            create_h5(root)
            c=make_config('spsn','shd'); c['data']['root']=root
            spec=c['data']; seed=57
            original=SHDEvents(Path(root)/'shd_train.h5')
            aug=DeterministicJitterAugmentedSubset(original,np.arange(4),20,1,seed+65537,time_scale_min=.9,time_scale_max=1.1)
            binner=SHDBinner(.0005,1.,700,1)
            ds=EventFrames(spec,True,seed)
            for i in range(8):
                expected,y=binner([aug[i]])
                x,label=ds[i]
                self.assertTrue(torch.equal(x,expected[0])); self.assertEqual(label,int(y[0]))
                self.assertTrue(torch.equal(x,ds[i][0]))
            plain=EventFrames(spec,False,seed)
            self.assertEqual(len(plain),4)
            x,_=plain[0]
            value=float(x.sum())
            empty=float(plain[1][0].sum())
            original.close()
            ds.source.dataset.close(); plain.source.dataset.close()
            # Original float32 division rounds 1.0/.0005 below 2000 on this path.
            # Match actual GateZO behavior, not an independently changed boundary rule.
            self.assertEqual(value,4.)
            self.assertEqual(empty,0.)
            self.assertFalse(list(Path(root).glob('*.npy')))

    def test_seed_defaults(self):
        result=subprocess.run([sys.executable,'run_experiments.py','--dry-run','--cpu','--datasets','spike'],cwd=ROOT,capture_output=True,text=True,check=True)
        data=json.loads(result.stdout)
        self.assertEqual(sorted(set(c['seed'] for c in data['tasks'])),SEEDS)
        from constants import METHODS
        self.assertEqual(len(data['tasks']),len(METHODS)*len(SEEDS))

    def test_missing_data_no_download(self):
        c=make_config('spsn','spike'); c['data']['root']='__nonexistent_pp3_data__'
        with self.assertRaises(FileNotFoundError):
            build_loaders(c)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
