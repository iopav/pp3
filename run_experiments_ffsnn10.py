"""Run the copied Spike FFSNN with 10 summed time windows using PP3's controller."""
import copy
from pathlib import Path
import constants
import run_experiments as controller


def main():
    baseline = constants.make_config('ffsnn', 'spike')
    original = (constants.METHODS, constants.DATASETS, constants.SMOKE,
                constants.make_config, controller.entrypoint)

    def config(method, dataset):
        if (method, dataset) != ('ffsnn10', 'spike'):
            raise ValueError('This entry supports only ffsnn10 on spike')
        result = copy.deepcopy(baseline)
        result.update(method='ffsnn10', backend='ffsnn_spike10_sum_v1',
            temporal_compression=dict(source_steps=400, target_steps=10,
                                      reduction='sum', window_width=40))
        result['model']['time_steps'] = 10
        return result

    constants.METHODS = {'ffsnn10': {'spike': copy.deepcopy(baseline['model'])}}
    constants.DATASETS = {'spike': copy.deepcopy(baseline['data'])}
    constants.SMOKE = dict(constants.SMOKE, steps=None, layers=2)
    constants.make_config = config
    controller.entrypoint = lambda _: Path(__file__).resolve().parent/'ffsnn_spike10/SNNFF_SPIKE10.py'
    try:
        return controller.main()
    finally:
        (constants.METHODS, constants.DATASETS, constants.SMOKE,
         constants.make_config, controller.entrypoint) = original


if __name__ == '__main__':
    raise SystemExit(main())
