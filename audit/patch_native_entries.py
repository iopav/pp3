"""One-time, checked edits of existing entry scripts; not a runtime launcher."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_ff():
    path = ROOT/'SNN_FF-main/FF_SNN/SNNFF_SHD.py'
    text = path.read_text(encoding='utf-8')
    assert 'PP3_CONFIG' not in text
    text = text.replace('import tonic\nimport tonic.transforms as transforms\nfrom tonic import DiskCachedDataset',
        "runtime = None\nif os.environ.get('PP3_CONFIG'):\n    from pp3.native_runtime import from_environment\n    runtime = from_environment()\nelse:\n    import tonic\n    import tonic.transforms as transforms\n    from tonic import DiskCachedDataset")
    text = text.replace('num_steps = 10', "num_steps = 10\nnum_classes, input_size = 20, 700\nif runtime:\n    settings, spec = runtime.config['model'], runtime.config['data']\n    beta, threshold = settings['beta'], settings['threshold']\n    num_steps = runtime.config.get('smoke_steps') or spec['steps']\n    num_classes, input_size = spec['classes'], spec['inputs']\n    # Reserve label channels when raw input has fewer channels than classes.\n    input_size = max(input_size, num_classes)")
    text = text.replace('linear_features=1000', 'linear_features=output_size')
    start = text.index("save_path = './data/SHD/baked/'")
    end = text.index('def superimpose_label', start)
    original = text[start:end]
    text = text[:start] + "if runtime:\n    train_x, train_y, test_x, test_y = runtime.arrays()\n    if input_size > spec['inputs']:\n        train_x = F.pad(train_x, (0, input_size-spec['inputs']))\n        test_x = F.pad(test_x, (0, input_size-spec['inputs']))\nelse:\n" + ''.join('    '+line+'\n' for line in original.splitlines()) + '\n' + text[end:]
    text = text.replace(':20] = 0', ':num_classes] = 0').replace('range(20)', 'range(num_classes)')
    text = text.replace('n_units = 500', "n_units = settings['hidden_size'] if runtime else 500\nif runtime:\n    runtime.begin('prepare', unknown='Native Python initialization arithmetic not audited')")
    text = text.replace('LeakyLayer(700, n_units)', 'LeakyLayer(input_size, n_units)')
    text = text.replace("train_acc_path = 'train_acc.txt'", "train_acc_path = str(runtime.output/'train_acc.txt') if runtime else 'train_acc.txt'")
    text = text.replace("test_acc_path = 'test_acc.txt'", "test_acc_path = str(runtime.output/'test_acc.txt') if runtime else 'test_acc.txt'")
    text = text.replace('torch.manual_seed(42)', "torch.manual_seed(runtime.config['seed'] if runtime else 42)")
    text = text.replace('learning_rate = 0.001', "learning_rate = settings['lr'] if runtime else 0.001")
    text = text.replace('num_epochs = 1 + (600)', "num_epochs = settings['epochs'] if runtime else 1 + (600)")
    text = text.replace('batch_size = 4096', "batch_size = settings['batch_size'] if runtime else 4096\nif runtime:\n    runtime.end()")
    text = text.replace('print_evaluation()\nfor epoch', 'if not runtime:\n    print_evaluation()\nfor epoch')
    text = text.replace('for epoch in range(num_epochs):', "for epoch in range(num_epochs):\n    if runtime:\n        runtime.begin('train', epoch+1, unknown='Native Adam Python scalar bias correction not counted')")
    text = text.replace('    test_rec(epoch)', "    if runtime:\n        runtime.end(loss=float(loss.detach()))\n        runtime.begin('test', epoch+1)\n    test_rec(epoch)\n    if runtime:\n        runtime.end(test_final_list['test_acc'][-1]/100)\n        runtime.epoch_row(epoch+1)\n        runtime.rows[-1]['train_accuracy'] = train_final_list['train_acc'][-1]/100\n        runtime._csv()")
    text = text.replace('if (epoch + 1) % 1 == 0:', 'if not runtime and (epoch + 1) % 1 == 0:')
    text += "\nif runtime:\n    runtime.finish(model.state_dict())\n"
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    patch_ff()
