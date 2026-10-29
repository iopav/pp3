import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'gate_zo_bdnn/train_shd_individual_pool_structural_zo.py'
# Exercise the production indexing helper without importing optional HDF5 I/O.
_tree = ast.parse((SOURCE.parent / 'train_shd_bdnn_rsnn_pure.py').read_text(encoding='utf-8'))
_selected = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selected_traces')
_selected.returns = None
for _arg in _selected.args.args:
    _arg.annotation = None
exec(compile(ast.Module(body=[_selected], type_ignores=[]), str(SOURCE), 'exec'))


class GrowthLifetimeTests(unittest.TestCase):
    def run_loop(self, prepared=False, eligibility=False, cleanup=True):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'round_index')
        if not cleanup:
            # A reference execution with identical arithmetic and no early releases.
            class RemoveDeletes(ast.NodeTransformer):
                def visit_Delete(self, node):
                    return ast.copy_location(ast.Pass(), node)
            loop = RemoveDeletes().visit(loop)
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(SOURCE), 'exec')
        refs, released, checkpoints = [], [], []
        labels = torch.tensor([0, 1, 0, 1])
        class State:
            def __init__(self, h): self.hidden_size = h
            def select_from_pool(self, old, selected): return State(old + len(selected))
            def checkpoint(self): return self.hidden_size
        def extract(loader, state, device, bins):
            if loader == 'train':
                released.append(all(r() is None for r in refs))
            traces = torch.arange(4*bins*state.hidden_size, dtype=torch.float32).reshape(4,bins,state.hidden_size)
            if loader == 'train': refs.append(weakref.ref(traces))
            return traces, labels.clone(), torch.zeros(state.hidden_size)
        core = NS(extract_readout_traces=extract, selected_traces=selected_traces)
        if eligibility:
            core.extract_readout_traces_with_eligibility = lambda loader,state,device,bins,*rest: (*extract(loader,state,device,bins), torch.ones(4,2))
        if prepared:
            core.prepare_structural_candidates = lambda loader,state,traces,labels,selected,old,*rest: dict(state=State(old+len(selected)), traces=selected_traces(traces,old,selected), record={'prepared': True})
        # First accept, then reject, then no selected candidates.
        selections = iter([[0], [1], []])
        accuracies = iter([.5, .4, .5])
        def solve(*args):
            acc = next(accuracies)
            return torch.ones(1), {'accuracy': acc}, {'accuracy': acc}, torch.zeros(1), torch.ones(1)
        runtime = NS(rows=[])
        runtime.begin = lambda *a: None
        runtime.end = lambda *a: None
        runtime.epoch_row = lambda *a,**k: runtime.rows.append({})
        runtime._csv = lambda: None
        env = dict(torch=torch,np=np,core=core,runtime=runtime,
            args=NS(max_growth_rounds=3,max_hidden=0,growth_step=1,seed=57,candidate_pool_size=2,input_scales=[],self_radii=[],old_to_new_scale=0,bias_scale=0,three_factor_mix=float(eligibility),readout_bins=2,readout_warmup_bins=0,temporal_windows=1,eligibility_tau=1,membrane_surrogate_width=1,selection_mode='gatezo',zo_sample_size=4,ridge=.5,growth_min_delta=.01,growth_failure_patience=9),
            start_round=1,state=State(0),train_loader='train',val_loader='val',test_loader='test',device='cpu',train_labels=labels,val_labels=labels,
            feature_mean=torch.zeros(1),feature_scale=torch.ones(1),beta=torch.zeros(1),train_features=torch.zeros(4,0),val_features=torch.zeros(4,0),
            current_val={'accuracy':0.},current_train={'accuracy':0.},history=[],failures=0,latest='unused',
            labels_by_position=labels,fit_pool=[],cal_pool=[],structural_partitions=None,proposal_pool=[],
            propose_individual_pool=lambda state,size,*a:(State(state.hidden_size+size),[{},{}]),
            select_residual_diverse_neurons=lambda *a:(next(selections),{}),
            internal_positions=lambda *a:([0,1],[2,3]),
            choose_new_neurons=lambda *a:([0],{}),
            sequence_features=lambda x,args:x.mean(1),solve_sequence_ridge=solve,
            apply_sequence_readout=lambda *a:torch.ones(4,2),
            blocks=NS(core=NS(classification_metrics=lambda *a:{'accuracy':.5})))
        with patch.object(torch,'save',side_effect=lambda obj,path:checkpoints.append(obj['state'])), contextlib.redirect_stdout(io.StringIO()):
            exec(code,env)
        return env, released, refs, checkpoints

    def test_pool_released_and_accept_rollback_empty_paths_preserved(self):
        for prepared, eligibility in ((False,False),(True,False),(False,True)):
            with self.subTest(prepared=prepared,eligibility=eligibility):
                a, released, refs, checkpoints = self.run_loop(prepared,eligibility)
                b, retained, _, reference_checkpoints = self.run_loop(prepared,eligibility,False)
                self.assertEqual(released,[True,True,True])
                self.assertEqual(retained,[True,False,False])
                self.assertTrue(all(r() is None for r in refs))
                self.assertEqual(checkpoints,[1,1,1])
                self.assertEqual(checkpoints,reference_checkpoints)
                self.assertEqual(a['history'],b['history'])
                for name in ('train_features','val_features','beta','feature_mean','feature_scale'):
                    torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
                self.assertNotIn('observed_traces',a)

if __name__ == '__main__':
    unittest.main()
