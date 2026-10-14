"""PP3 bridge to the original EfficientTrain++ ResNet50 curriculum.

The resolution list comes from the upper-level entry. Epoch rounding and
resolution-dependent LR coordinates match that entry and auto_load_model.
The image bridge is explicit: train-fitted range -> 8-bit RGB -> original
build_transform (including RandAugment and PIL Lanczos low-pass resizing).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROTOCOL = 'efficienttrain_pp_resnet50_v1'


def stage_plan(resolutions, budget_epochs, base_size=224, smoke=False):
    if len(resolutions) != 10 or budget_epochs < 1 or base_size < 32:
        raise ValueError('ET requires ten stages, a positive budget and base size >= 32')
    if not smoke and (budget_epochs < 10 or base_size != 224):
        raise ValueError('Full ET ResNet50 requires --epochs >= 10 and input_size=224; use --smoke for a tiny complete curriculum')
    stages = []
    previous_end, previous_size = 0, None
    for index, original_size in enumerate(resolutions):
        size = max(32, round(original_size * base_size / 224)) if smoke else original_size
        # Preserve the original arithmetic order: (base/size)**2 can round
        # 196 down to 195 before int(), changing the official schedule.
        schedule_epochs = int(budget_epochs * ((base_size * base_size) / (size * size)))
        start = round(previous_end * previous_size ** 2 / size ** 2) if previous_size else 0
        end = int(schedule_epochs / 10 * (index + 1))
        if smoke:
            # One complete pass per stage; not a performance/budget experiment.
            schedule_epochs, start, end = 10, index, index + 1
        if end <= start:
            raise ValueError(f'Empty ET stage {index}; increase the epoch budget')
        stages.append(dict(stage=index, input_size=size,
                           aa=f'rand-m{index}-mstd0.5-inc1',
                           schedule_epochs=schedule_epochs, start_epoch=start,
                           end_epoch=end, epochs=end-start))
        previous_end, previous_size = end, size
    return stages


def epoch_plan(stages):
    return [dict(stage, schedule_epoch=epoch)
            for stage in stages for epoch in range(stage['start_epoch'], stage['end_epoch'])]


def schedules(stages, steps, args, cosine_scheduler):
    """Concatenate the slices visited by the original stage/resume loop."""
    learning_rates, decays = [], []
    for stage in stages:
        epochs = stage['schedule_epochs']
        warmup = min(args.warmup_epochs, epochs - 1)
        lr = cosine_scheduler(args.lr, args.min_lr, epochs, steps, warmup_epochs=warmup)
        wd = cosine_scheduler(args.weight_decay, args.weight_decay_end, epochs, steps)
        left, right = stage['start_epoch'] * steps, stage['end_epoch'] * steps
        learning_rates.append(lr[left:right])
        decays.append(wd[left:right])
    return np.concatenate(learning_rates), np.concatenate(decays)


def training_range(source):
    """Fit a single amplitude scale using training inputs only (not labels/test)."""
    low, high = float('inf'), float('-inf')
    for index in range(len(source)):
        x, _ = source[index]
        if not torch.isfinite(x).all():
            raise ValueError('Non-finite ET input')
        low, high = min(low, float(x.min())), max(high, float(x.max()))
    if not np.isfinite([low, high]).all():
        raise ValueError('Empty ET training data')
    return low, high


class CurriculumImages(Dataset):
    def __init__(self, source, base_size, bounds, transform):
        self.source, self.base_size = source, base_size
        self.bounds, self.transform = bounds, transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, label = self.source[index]
        x = torch.nn.functional.interpolate(x[None, None], size=(self.base_size, self.base_size),
                                            mode='bilinear', align_corners=False)[0, 0]
        low, high = self.bounds
        # One training-fitted scale preserves between-example amplitude differences.
        # Signed values are mapped, not zeroed; quantization and test-range clipping
        # are an explicit representation change required by the native PIL pipeline.
        x = ((x-low) / (high-low) if high > low else torch.zeros_like(x))
        pixels = x.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        return self.transform(Image.fromarray(pixels).convert('RGB')), label


def launch(resolutions):
    from main_buffer import get_args_parser, main
    config = (json.loads(os.environ['PP3_CONFIG_JSON']) if os.environ.get('PP3_CONFIG_JSON')
              else json.loads(Path(os.environ['PP3_CONFIG']).read_text(encoding='utf-8')))
    settings = config['model']
    if settings.get('model') != 'resnet50':
        raise ValueError('This upper ET recipe is specifically for ResNet50')
    if settings.get('protocol') != PROTOCOL or settings.get('image_encoding') != 'train_range_uint8':
        raise ValueError('ET requires the current explicit protocol and image encoding configuration')
    args = get_args_parser().parse_args([])
    args.pp3_et_stages = stage_plan(resolutions, settings['epochs'], settings['input_size'], config.get('smoke', False))
    args.pp3_et_protocol = PROTOCOL
    main(args)
