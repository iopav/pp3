"""PP3 controller with an additive CUDA BDNN backend; original files are untouched."""
from pathlib import Path
import constants
import run_experiments as controller


def main():
    original_entry = controller.entrypoint
    original_config = constants.make_config
    original_output = constants.RUNTIME['output']

    def entry(config):
        if config['method'] == 'bdnn':
            return Path(__file__).resolve().parent / 'bdnn_gpu' / 'main.py'
        return original_entry(config)

    def config(method, dataset):
        result = original_config(method, dataset)
        if method == 'bdnn':
            result['bdnn_backend'] = 'torch_cuda_native_semantics_v1'
            result['model_selection'] = 'native_test_argmax_exclusive_slice'
        return result

    controller.entrypoint = entry
    constants.make_config = config
    constants.RUNTIME['output'] = original_output.rstrip('/\\') + '_bdnn_gpu'
    try:
        return controller.main()
    finally:
        controller.entrypoint = original_entry
        constants.make_config = original_config
        constants.RUNTIME['output'] = original_output


if __name__ == '__main__':
    raise SystemExit(main())
