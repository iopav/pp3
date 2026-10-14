import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
ET = ROOT / 'EfficientTrain-main'
sys.path.insert(0, str(ET))
from pp3_et import stage_plan, epoch_plan, schedules, training_range, CurriculumImages
from datasets_buffer import build_transform
with patch.dict(os.environ, {'PP3_CONFIG': 'unit-test-import-only'}):
    from main_buffer import get_args_parser, main
from utils_buffer import cosine_scheduler


class ETCurriculumTests(unittest.TestCase):
    def test_official_budget_and_resume_rounding(self):
        sizes = [96]*2 + [160]*4 + [224]*4
        for budget in (10, 100, 200):
            stages = stage_plan(sizes, budget)
            previous_end, previous_size = 0, 224
            for index, stage in enumerate(stages):
                # Original upper entry + utils_buffer.auto_load_model formulas.
                total = int(budget * ((224*224)/(sizes[index]*sizes[index])))
                end = int(total / 10 * (index+1))
                start = round(previous_end * previous_size**2 / sizes[index]**2)
                self.assertEqual((stage['schedule_epochs'], stage['start_epoch'], stage['end_epoch']),
                                 (total, start, end))
                previous_end, previous_size = end, sizes[index]
            self.assertEqual({s['stage'] for s in epoch_plan(stages)}, set(range(10)))
        self.assertEqual(len(epoch_plan(stage_plan(sizes, 100))), 226)
        with self.assertRaises(ValueError):
            stage_plan(sizes, 1)

    def test_lr_slices_follow_original_resume_coordinates(self):
        stages = stage_plan([96]*2+[160]*4+[224]*4, 100)
        args = SimpleNamespace(lr=.004, min_lr=1e-6, warmup_epochs=20,
                               weight_decay=.05, weight_decay_end=.05)
        lr, wd = schedules(stages, 2, args, cosine_scheduler)
        offset = 0
        for stage in stages:
            reference = cosine_scheduler(.004, 1e-6, stage['schedule_epochs'], 2, warmup_epochs=20)
            count = stage['epochs']*2
            np.testing.assert_array_equal(lr[offset:offset+count],
                reference[stage['start_epoch']*2:stage['end_epoch']*2])
            offset += count
        self.assertEqual(len(lr), len(epoch_plan(stages))*2)
        np.testing.assert_allclose(wd, .05)

    def test_native_transform_and_worker_stage_change(self):
        args = get_args_parser().parse_args([])
        args.base_input_size = 64
        source = TensorDataset(torch.linspace(-2, 3, 4*16*9).reshape(4, 16, 9), torch.arange(4))
        bounds = training_range(source)
        self.assertEqual(bounds, (-2., 3.))
        args.input_size, args.aa = 32, 'rand-m0-mstd0.5-inc1'
        dataset = CurriculumImages(source, 64, bounds, build_transform(True, args))
        text = str(dataset.transform)
        self.assertIn('RandAugment', text)
        self.assertIn('lanczos', text.lower())
        loader = DataLoader(dataset, batch_size=2, num_workers=1, persistent_workers=False)
        self.assertEqual(tuple(next(iter(loader))[0].shape), (2, 3, 32, 32))
        args.input_size, args.aa = 64, 'rand-m9-mstd0.5-inc1'
        dataset.transform = build_transform(True, args)
        self.assertEqual(tuple(next(iter(loader))[0].shape), (2, 3, 64, 64))
        # Evaluation uses training bounds, remains deterministic and does not augment.
        dataset.transform = build_transform(False, args)
        first, second = dataset[0][0], dataset[0][0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (3, 64, 64))

    def test_full_upper_entry_is_selected(self):
        import run_experiments
        self.assertEqual(run_experiments.entrypoint(dict(method='efficienttrain', dataset='spike')).name,
                         '_ET_pp_main_res50.py')

    def test_direct_fixed_size_entry_fails_instead_of_mislabeling_baseline(self):
        with patch.dict(os.environ, {'PP3_CONFIG': 'unused'}):
            with self.assertRaisesRegex(ValueError, 'must launch through'):
                main(get_args_parser().parse_args([]))


if __name__ == '__main__':
    unittest.main()
