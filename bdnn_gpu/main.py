"""PP3 CUDA port retaining native classification control flow and final slicing."""
import json
import torch
from pp3.native_runtime import write_json
from bdnn_gpu.model import BDNNGPU
from bdnn_gpu.runtime import GPURuntime


def accuracy(target, prediction):
    return float((target == prediction).all(1).double().mean())


@torch.no_grad()
def main():
    runtime = GPURuntime()
    if runtime.device.type != 'cuda':
        raise RuntimeError('This entry requires CUDA; original CPU entry remains available')
    settings = runtime.config['model']
    x, labels, xt, labels_t = runtime.arrays()
    x, xt = x.to(runtime.device), xt.to(runtime.device)
    classes = runtime.config['data']['classes']
    target = torch.nn.functional.one_hot(labels.to(runtime.device), classes).float()
    test_target = torch.nn.functional.one_hot(labels_t.to(runtime.device), classes).float()
    model = BDNNGPU(x.shape[2], classes, settings, runtime.device)
    runtime.model = model
    error, rate, rate_new, rate_history, test_history = 30., 0., 1/15, [0.], []
    epoch = 0
    while model.L < settings['epochs'] and error > settings['tol'] and rate_new - rate >= .001:
        epoch += 1
        runtime.begin('train', epoch, excluded='Python scalar loop/control arithmetic')
        if model.L != 0 and model.L % 50 == 0:
            rate, rate_new = rate_new, max(rate_history)
            rate_history = []
        with runtime.counter.section('candidate_search'):
            candidate = model.sc_Search(x, target)
        if candidate is None:
            runtime.end()
            runtime.epoch_row(epoch, unit='growth_round_failed')
            break
        model.addNodes(*candidate)
        with runtime.counter.section('readout_fit_and_training_loss'):
            _, cost = model.upgradeSCN(x, target)
        error = float(cost)
        with runtime.counter.section('train_evaluation'):
            prediction = model.getLabel(x)
            last_train = accuracy(target, prediction)
        result = runtime.end(last_train, error)
        runtime.begin('test', epoch)
        prediction_t = model.getLabel(xt)
        last_train = accuracy(target, prediction)
        rate_history.append(last_train)
        last_test = accuracy(test_target, prediction_t)
        runtime.end(last_test)
        runtime.epoch_row(epoch)
        runtime.rows[-1].update({'train_stage_'+stage+'_flops': value
            for stage, value in result['known_flops_by_stage'].items()})
        runtime._csv()
        test_history.extend([last_test] * (1 if not test_history else candidate[0].shape[1]))
        print(f'round={epoch} nodes={model.W.shape[1]} train={last_train:.6f} test={last_test:.6f}', flush=True)
    runtime.begin('final_refit')
    history = torch.tensor(test_history, dtype=torch.float64, device=runtime.device)
    # Keep empty-history failure, test selection, and exclusive index unchanged.
    best = history.max()
    index = int(history.argmax())
    model.W, model.U = model.W[:, :index], model.U[:index, :index]
    h, _ = model.getH(x)
    model.computeBeta(h, target)
    predicted = model.getLabel(xt)
    final_accuracy = accuracy(test_target, predicted)
    runtime.end(final_accuracy)
    runtime.finish({key: value.cpu().numpy() for key, value in model.state_dict().items()})
    path = runtime.output / 'status.json'
    status = json.loads(path.read_text(encoding='utf-8'))
    status.update(backend='torch_cuda_native_semantics_v1', test_used_for_selection=True,
        final_slice_index=index, final_test_accuracy=final_accuracy,
        parameter_bytes=model.parameter_bytes(), auxiliary_mem_bytes=0,
        state_devices={key: str(value.device) for key, value in model.state_dict().items()})
    write_json(path, status)


if __name__ == '__main__':
    main()
