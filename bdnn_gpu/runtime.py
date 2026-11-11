"""Extend PP3 observations without altering the original runtime."""
import torch
from pp3.native_runtime import Runtime, write_json


class GPURuntime(Runtime):
    def begin(self, *args, **kwargs):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        super().begin(*args, **kwargs)
        self.model.counter = self.counter
        self.counter.mark_excluded('rng_engine', 'Native NumPy RNG engine; normal transform estimate and CUDA affine arithmetic counted')

    def end(self, *args, **kwargs):
        result = super().end(*args, **kwargs)
        self.model.counter = None
        result['peak_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated(self.device)
            if self.device.type == 'cuda' else None)
        result['peak_cuda_reserved_bytes'] = (torch.cuda.max_memory_reserved(self.device)
            if self.device.type == 'cuda' else None)
        write_json(self.output/f'epoch_{self.epoch:04d}_{self.phase}_operations.json', result)
        return result

    def epoch_row(self, epoch, unit='growth_round'):
        super().epoch_row(epoch, unit)
        row = self.rows[-1]
        row.update(para_mem_bytes=self.model.parameter_bytes(), optimizer_state_bytes=0,
                   auxiliary_mem_bytes=0, hidden_state_bytes=0, **{'hiddenstatememfor bp': 0})
        for phase in ('train', 'test'):
            for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
                row[f'{phase}_{key}'] = self.results.get((epoch, phase), {}).get(key)
        self._csv()
