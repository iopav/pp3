from utils import train
from data import load_classwise_PMNIST, load_classwise_NMNIST, load_SHD
from model import EchoSpike
import torch
import pickle
import os

# hyperparameters
class Args:
    def __init__(self):
        self.model_name = 'test'
        self.dataset = 'nmnist'
        self.online = True
        self.device = 'cuda'
        self.recurrency_type = 'dense'
        self.lr = 1e-4
        self.epochs = 1000
        self.augment = False
        self.batch_size = 128 # 64 saccade and 64 predictive before weight update -> 128
        self.n_hidden = 3*[200]
        if self.dataset == 'nmnist':
            self.c_y = [1e-4, -1e-4] if not self.online else [2, -1]
            self.inp_thr = 0.02
            self.n_inputs = 2*34*34
            self.n_outputs = 10
            self.n_time_bins = 10
            self.beta = 0.9
        elif self.dataset == 'pmnist':
            self.c_y = [1e-4, -1e-4]
            self.inp_thr = 0.0
            self.n_inputs = 28*28
            self.n_outputs = 10
            self.poisson_scale = 0.8
            self.n_time_bins = 10
            self.beta = 0.9
        elif self.dataset == 'shd':
            self.inp_thr = 0.05
            self.c_y = [8e-4, -4e-4] if not self.online else [1.5, -1.5]
            self.n_inputs = 700
            self.n_outputs = 20
            self.n_time_bins = 100
            self.beta = [0.94,0.96,0.98,1.0]#-0.95
        else:
            raise NotImplementedError

if __name__ == '__main__':
    args = Args()
    runtime = None
    if os.environ.get('PP3_CONFIG'):
        from pp3.native_runtime import from_environment
        from data import classwise_loader
        runtime = from_environment()
        config, settings, spec = runtime.config, runtime.config['model'], runtime.config['data']
        args.dataset, args.device = config['dataset'], str(runtime.device)
        args.n_inputs, args.n_outputs = spec['inputs'], spec['classes']
        args.n_time_bins = config.get('smoke_steps') or spec['steps']
        args.n_hidden = [settings['hidden_size']]*settings['layers']
        for key in ('online', 'lr', 'epochs', 'batch_size', 'beta', 'recurrency_type', 'c_y', 'inp_thr'):
            setattr(args, key, settings[key])
        common_train, common_test = runtime.loaders()
        train_loader = classwise_loader(common_train.dataset, common_train.dataset.targets,
            args.n_outputs, args.batch_size, seed=config['seed'])
        test_loader = classwise_loader(common_test.dataset, common_test.dataset.targets,
            args.n_outputs, args.batch_size, seed=config['seed'])
        train_loader._pp3_runtime = test_loader._pp3_runtime = runtime
        if any(len(pool)==0 for pool in train_loader.target_indeces):
            raise ValueError('ESPP native sampler requires at least one training example per class')
        os.chdir(runtime.output)
        os.makedirs('models', exist_ok=True)
    torch.manual_seed(config['seed'] if runtime else 123)
    # load dataset
    if runtime:
        pass
    elif args.dataset == 'nmnist':
        train_loader, _, test_loader = load_classwise_NMNIST(args.n_time_bins, split_train=True, batch_size=args.batch_size) 
    elif args.dataset == 'pmnist':
        train_loader, test_loader = load_classwise_PMNIST(args.n_time_bins, scale=args.poisson_scale, batch_size=args.batch_size)
    elif args.dataset == 'shd':
        train_loader, test_loader = load_SHD(batch_size=args.batch_size)

    # train model
    if runtime:
        runtime.begin('prepare', unknown='Native initialization Python arithmetic not audited')
    SNN = EchoSpike(args.n_inputs, args.n_hidden, c_y= args.c_y, beta=args.beta,
                     device=args.device, recurrency_type=args.recurrency_type,
                     n_time_steps=args.n_time_bins, online=args.online, inp_thr=args.inp_thr).to(args.device)
    SNN.reset(0) # after sending to cuda the espp_layers' mem is not on cuda, resetting fixes that
    if runtime:
        runtime.end()
    
    loss_hist = train(SNN, train_loader, args.epochs, args.device, args.model_name,
                            batch_size=args.batch_size, online=args.online, lr=args.lr, augment=args.augment,
                            runtime=runtime, args=args, testloader=test_loader)
    if runtime:
        runtime.finish(dict(backbone=SNN.state_dict(),
            readouts=[p.state_dict() for p in runtime.readouts]))

    # Save the model, loss history and arguments
    torch.save(SNN.state_dict(), f'models/{args.model_name}.pt')
    torch.save(loss_hist, f'models/{args.model_name}_loss_hist.pt')
    with open(f'models/{args.model_name}_args.pkl', 'wb') as f:
        pickle.dump(args, f)
