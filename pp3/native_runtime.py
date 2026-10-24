"""PP3 原生实验的公共运行支持：配置、数据访问、分阶段计量和结果保存。

模型结构和学习算法由各项目入口实现，通过 PP3_CONFIG 显式接入本模块。
train、test 等阶段分别计量；epoch 总量再累加同一 epoch 下的所有已记录阶段。
阶段边界由调用方指定，本模块不会根据数据来自训练集还是测试集自动分类。
未知算子会保留诊断记录并使 FLOPs 覆盖状态不完整，不会因此中断训练。
本模块记录的 model_bytes 是序列化文件大小，不测量运行时辅助内存或显存峰值。
"""
import atexit
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from contextlib import contextmanager

import numpy as np
import torch
from .data import build_loaders
from .energy import Energy
from .flops import FlopCounter


def write_json(path, value):
    """以 UTF-8 保存 JSON；拒绝 NaN/Infinity，避免产生无效的数值报告。"""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   default=str, allow_nan=False), encoding='utf-8')


def check_finite_state(value):
    """递归检查待保存状态中的 Tensor/NumPy 数组，发现非有限值立即报错。"""
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError('Final model contains non-finite tensors')
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise FloatingPointError('Final model contains non-finite arrays')
    elif isinstance(value, dict):
        for item in value.values():
            check_finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite_state(item)


class MeasuredLoader:
    """取批次及可选设备搬运时暂停 FLOPs 计数，但不暂停阶段计时和能耗。

    data_wait_seconds 记录这些操作的主机侧耗时，不会从阶段总耗时中扣除；
    它也不是单独同步测得的 GPU 传输时间。yield 后模型计算恢复原计数状态。
    """
    def __init__(self, loader, runtime, device=None):
        self.loader, self.runtime, self.device = loader, runtime, device
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # 创建迭代器也可能触发采样或初始化，需与取批次一样排除其 FLOPs。
        counter = self.runtime.counter
        paused = counter.paused if counter else False
        if counter:
            counter.paused = True
        tick = time.perf_counter()
        iterator = iter(self.loader)
        self.runtime.data_wait += time.perf_counter() - tick
        if counter:
            counter.paused = paused
        while True:
            counter = self.runtime.counter
            paused = counter.paused if counter else False
            if counter:
                counter.paused = True
            tick = time.perf_counter()
            try:
                x, y = next(iterator)
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)
            except StopIteration:
                return
            finally:
                # 正常取批次、迭代结束或异常退出时，都恢复进入前的暂停状态。
                self.runtime.data_wait += time.perf_counter() - tick
                if counter:
                    counter.paused = paused
            yield x, y


class Runtime:
    """管理一个实验进程；同一时刻只允许一个 begin/end 计量阶段。"""

    def __init__(self):
        # 控制器通常通过环境中的 JSON 传配置，也兼容配置文件路径。
        self.config = (json.loads(os.environ['PP3_CONFIG_JSON']) if os.environ.get('PP3_CONFIG_JSON')
                       else json.loads(Path(os.environ['PP3_CONFIG']).read_text(encoding='utf-8')))
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # 严格启用确定性算法；遇到不支持的算子报错，不退化为仅警告。
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.output = Path(self.config['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.config['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.set_num_threads(self.config['threads'])
        seed = self.config['seed']
        # 使用当前实验配置中的 seed；本模块不把所有实验强制改为同一个 seed。
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rows, self.results = [], {}
        # rows 保存逐 epoch CSV 行；results 以 (epoch, phase) 保存阶段结果。
        self.counter = self.energy = None
        self.data_wait = 0.
        self.complete = True
        self.unknown_records = []
        self.finished = False
        self.error = None
        previous_hook = sys.excepthook
        # 保留未捕获异常的堆栈，退出时写入失败状态，同时保留原异常输出。
        def hook(kind, value, tb):
            self.error = ''.join(traceback.format_exception(kind, value, tb))
            previous_hook(kind, value, tb)
        sys.excepthook = hook
        atexit.register(self._exit)

    def loaders(self, device=False):
        """构建训练/测试加载器，包装数据计量边界；device=True 时逐批搬运。"""
        train, test, seconds = build_loaders(self.config)
        destination = self.device if device else None
        return MeasuredLoader(train, self, destination), MeasuredLoader(test, self, destination)

    def arrays(self):
        """为原生整数组算法一次性堆叠数据，保留其整数据驻留 RAM 的要求。"""
        tick = time.perf_counter()
        train, test = self.loaders()
        values = []
        for loader in (train, test):
            dataset=loader.dataset
            if loader is train and self.config['method']=='gatezo':
                # 合并 GateZO 训练与验证子集，并记录合并后的索引边界供调用方划分。
                validation=dataset.validation_dataset
                self.outer_array_indices=(np.arange(len(dataset)),
                    np.arange(len(dataset),len(dataset)+len(validation)))
                dataset=torch.utils.data.ConcatDataset([dataset,validation])
            # 按数据集索引取样，不消耗训练加载器的 shuffle 采样器状态。
            pairs = [dataset[i] for i in range(len(dataset))]
            values.extend((torch.stack([p[0] for p in pairs]),
                           torch.tensor([p[1] for p in pairs], dtype=torch.long)))
        return tuple(values)

    def ordered_loader(self, dataset, device=False):
        """构建不打乱顺序的加载器，适用于评估或要求标签顺序一致的特征提取。"""
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['model']['batch_size'],
            shuffle=False, num_workers=self.config['workers'],
            generator=torch.Generator().manual_seed(self.config['seed']))
        return MeasuredLoader(loader, self, self.device if device else None)

    @contextmanager
    def preprocessing(self):
        """显式标记预处理：仅暂停 FLOPs，耗时和能耗仍属于当前阶段。"""
        counter = self.counter
        paused = counter.paused if counter else False
        if counter:
            counter.paused = True
        tick = time.perf_counter()
        try:
            yield
        finally:
            self.data_wait += time.perf_counter()-tick
            if counter:
                counter.paused = paused

    def begin(self, phase, epoch=0, excluded=None):
        """开启独立阶段的 FLOPs 与能耗计量；禁止嵌套，须先 end 再 begin。

        phase 通常为 train、test、readout_fit、prepare 或 final_*。
        test 中是否包含训练集报告评估，取决于各方法入口安排的调用范围。
        excluded 仅登记调用方声明的排除项，不会自动剔除某段模型计算。
        """
        if self.counter is not None:
            raise RuntimeError('Nested measurement phases are not permitted')
        self.phase, self.epoch, self.data_wait = phase, epoch, 0.
        self.counter = FlopCounter(enabled=self.config.get('count_flops', True))
        self.counter.stage = phase
        if excluded:
            self.counter.mark_excluded('python_scalar_bookkeeping', excluded)
        self.energy = Energy(self.device, self.config['power_sample_seconds'])
        self.energy.__enter__()
        self.counter.__enter__()

    def end(self, accuracy=None, loss=None):
        """关闭当前阶段并保存计量结果；准确率和损失由调用方计算后传入。"""
        counter, energy = self.counter, self.energy
        if counter is None:
            raise RuntimeError('No active measurement phase')
        counter.__exit__(None, None, None)
        self.counter = None
        energy.__exit__(None, None, None)
        result = counter.summary()
        self.unknown_records.extend(dict(epoch=self.epoch, phase=self.phase, **item)
                                    for item in result['unknown'])
        calls = {}
        for item in self.unknown_records:
            calls[item['operator']] = calls.get(item['operator'], 0) + item['calls']
        # 每个阶段结束后原子替换未知算子诊断文件，累计保留整个实验的缺失规则。
        temporary = self.output/'flops_unknown.json.tmp'
        write_json(temporary, dict(calls_by_operator=calls, records=self.unknown_records))
        temporary.replace(self.output/'flops_unknown.json')
        self.complete = not self.unknown_records
        # 关闭计数时，不把完整 FLOPs 填成 0；None 表示未提供该指标。
        if not counter.enabled:
            result.update(complete=None, flops=None)
        result.update(energy.result(), data_wait_seconds=self.data_wait,
                      accuracy=None if accuracy is None else float(accuracy),
                      loss=None if loss is None else float(loss))
        for key in ('accuracy', 'loss'):
            if result[key] is not None and not np.isfinite(result[key]):
                raise FloatingPointError('Non-finite '+key)
        # 相同 (epoch, phase) 再次结束会覆盖旧记录，不会自动合并多个计量片段。
        self.results[(self.epoch, self.phase)] = result
        return result

    def epoch_row(self, epoch, unit='epoch'):
        """输出已统计的阶段 FLOPs 与 epoch 合计；FFT/other 已包含在总数内。"""
        row = dict(epoch=epoch)
        phases = ['train', 'test'] + [phase for number, phase in self.results
            if number == epoch and phase not in ('train', 'test')]
        # 保留 train_flops/test_flops、train_gpu_energy_j/test_gpu_energy_j 等独立列。
        # 缺少的阶段填空；其他阶段（如 readout_fit）按实际记录追加。
        for phase in phases:
            result = self.results.get((epoch, phase), {})
            for key in ('accuracy', 'loss', 'seconds', 'gpu_energy_j',
                        'other_flops_estimated'):
                row[f'{phase}_{key}'] = result.get(key)
            row[f'{phase}_flops'] = result.get('known_flops')
            row[f'{phase}_fft_flops_included'] = result.get('fft_flops_estimated')
        measured = [self.results[(epoch, phase)] for phase in phases if (epoch, phase) in self.results]
        # 合计包含本 epoch 的全部已记录阶段，不等于纯训练成本。
        # known_flops 已含 FFT 和 other 估算项，不能再把分项加一次。
        # 即使存在未知算子也输出已统计的合计；未知算子仍记录于 flops_unknown.json。
        row['epoch_flops'] = sum(r['known_flops'] for r in measured)
        row['epoch_seconds'] = sum(r['seconds'] for r in measured)
        # 任一已记录阶段的能耗不可用，合计也填空，不能把缺失能耗当成 0。
        row['epoch_gpu_energy_j'] = (sum(r['gpu_energy_j'] for r in measured)
            if measured and all(r['gpu_energy_j'] is not None for r in measured) else None)
        self.rows.append(row)
        self._csv()

    def _csv(self):
        """重写当前实验的原始 metrics.csv；字段取所有已记录行的键并集。"""
        if not self.rows:
            return
        fields = list(dict.fromkeys(k for row in self.rows for k in row))
        with (self.output/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self, state):
        """检查并保存最终模型状态，补齐原始指标并写出完成性报告。"""
        check_finite_state(state)
        path = self.output/'model.pt'
        torch.save(state, path)
        # final_model_bytes/model_bytes 是 model.pt 文件大小，包含序列化开销。
        # 它既不是参数张量字节数之和，也不是训练期间的内存或显存峰值。
        for row in self.rows:
            row['final_model_bytes'] = path.stat().st_size
        self._csv()
        self.finished = True
        # prepare/final_* 的一次性开销单独存入 status，供 reporting 单独导出。
        # 调用方通常让这些阶段使用 epoch=0，避免与常规 epoch 合计重叠。
        # 未知算子的完整记录已由 end() 写入 flops_unknown.json。
        one_time = [dict(epoch=number, phase=phase, flops=result.get('known_flops'),
                    **{k: result.get(k) for k in ('gpu_energy_j','seconds','accuracy')})
                    for (number, phase), result in self.results.items()
                    if phase == 'prepare' or phase.startswith('final_')]
        write_json(self.output/'status.json', dict(
            artifact_version=3, experiment_id=self.config.get('experiment_id'),
            method=self.config['method'], dataset=self.config['dataset'], seed=self.config['seed'],
            one_time_metrics=one_time, flops_unknown_file='flops_unknown.json',
            status='success' if self.complete else 'flops_incomplete',
            training_completed=True, flops_complete=self.complete if self.config.get('count_flops', True) else None,
            model_bytes=path.stat().st_size, epochs=len(self.rows), actual_device=str(self.device)))

    def _exit(self):
        """正常解释器退出时尽量关闭计量；未完成或存在未捕获异常则标记失败。

        强制终止进程时不保证执行此回调，控制器还需负责记录退出状态。
        """
        if self.counter is not None:
            try:
                self.end()
            except Exception:
                pass
        if not self.finished or self.error:
            write_json(self.output/'status.json', dict(status='failed',
                training_completed=False, reason=self.error or 'Original entry did not finish'))


def from_environment():
    """仅在设置 PP3_CONFIG 时接入；未接入的原始项目入口保持独立运行。"""
    return Runtime() if os.environ.get('PP3_CONFIG') else None
