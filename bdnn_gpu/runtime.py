"""Extend PP3 observations without altering the original runtime."""
from pp3.native_runtime import Runtime


class GPURuntime(Runtime):
    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        if self.config['dataset'] == 'shd':
            from .fast_counter import RepeatedOpCounter
            previous = self.counter
            previous.__exit__(None, None, None)
            self.counter = RepeatedOpCounter(previous.enabled)
            self.counter.records.update(previous.records)
            self.counter.__enter__()
        self.model.counter = self.counter
        self.counter.mark_excluded('rng_engine', 'Native NumPy RNG engine; normal transform estimate and CUDA affine arithmetic counted')

    def end(self, *args, **kwargs):
        result = super().end(*args, **kwargs)
        self.model.counter = None
        return result
