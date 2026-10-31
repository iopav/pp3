"""Edit paths and method/dataset settings here before submitting to the cluster."""
import copy
import json
import os
from pathlib import Path

# Shared defaults for full experiment runs.
EPOCHS = 100
BATCH_SIZE = 128

DATA_ROOT = Path(os.environ.get('PP3_DATA_ROOT', '/home/yizhil/pp3/data'))
DATASETS = {
    'spike': dict(root=str(DATA_ROOT/'spike'), steps=400, inputs=64, classes=20,
                  train_x='X_train_20_cla_400steps.npy', train_y='Y_train_20_cla_400steps.npy',
                  test_x='X_test_20_cla_400steps.npy', test_y='Y_test_20_cla_400steps.npy'),
    'texture': dict(root=str(DATA_ROOT/'texture'), steps=3775, inputs=9, classes=12,
                    train_x='X_train_texture.npy', train_y='Y_train_texture.npy',
                    test_x='X_test_texture.npy', test_y='Y_test_texture.npy'),
    'shd': dict(root=str(DATA_ROOT/'SHD'), steps=2000, inputs=700, classes=20,
                dt=.0005, max_time=1., channel_group_size=1,
                jitter_std=20., jitter_copies=1, time_scale_min=.90, time_scale_max=1.10),
}
SPSN = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, hidden_size=128, layers=3, lr=.001,
            tau_mem=.02, tau_syn=.02, time_step=.001, spike_mode='SB',
            loss_mode='mean', reg_thr=0.)
TP = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, hidden_size=450, layers=1, lr=.0001,
          layerwise=True, l_vth=1., l_leak_m=.96, l_leak_t=.97,
          l_rst_type='soft', l_out_leak_m=1., l_rec=False, norm_type='none',
          surrogate_type='1', surrogate_scale=1., train_s=False)
METHODS = {m: {d: copy.deepcopy(p) for d in DATASETS} for m,p in [('spsn',SPSN),('tp',TP)]}
EXTRA_METHODS = {
    'gatezo': dict(epochs=EPOCHS,batch_size=BATCH_SIZE,hidden_size=0,layers=1,pool_size=512,growth_step=16,
        zo_min_keep=8,zo_sparsity_lambda=.000025,zo_sample_size=2048,ridge=.5,
        growth_failure_patience=12,growth_min_delta=.002,diversity_strength=.7,
        readout_bins=100,readout_warmup_bins=5,temporal_windows=16,
        input_scales=[2,3,4,6],self_radii=[.7,.9,1.1,1.3],old_to_new_scale=.5,bias_scale=.2,
        candidate_min_rate=.0005,candidate_max_rate=.30,min_candidate_score=0.,
        validation_ratio=.2,validation_speakers=2,zo_calibration_ratio=.2,
        dt=.0005,tau_syn=.010,tau_mem=.020,input_branches=4,
        input_delay_ms=[0.,10.,25.,50.,100.,200.],frequency_bandwidths=[32,64,128,256],
        heterogeneous_tau_fraction=.5,tau_syn_min=.005,tau_syn_max=.025,tau_mem_min=.015,tau_mem_max=.060),
    'ffsnn': dict(epochs=EPOCHS,batch_size=BATCH_SIZE,hidden_size=500,layers=2,lr=.001,beta=.9,threshold=5.,loss_alpha=4.),
    'espp': dict(epochs=EPOCHS,batch_size=BATCH_SIZE,hidden_size=200,layers=3,lr=.0001,
                 beta=[.94,.96,.98,1.0],online=True,recurrency_type='dense',c_y=[1.5,-1.5],inp_thr=.05,
                 readout_lr=.001,readout_epochs=1),
    'bdnn': dict(epochs=EPOCHS,batch_size=BATCH_SIZE,hidden_size=100,layers=1,pool_size=100,growth_step=3,
                 alpha=.13,beta=1.,lambdas=[.5,1,3,5,7,9,15,25,50,100,150,200],
                 r=[.99,.999,.9999,.99999,.999999],tol=.0005),
}
for method,settings in EXTRA_METHODS.items():
    METHODS[method]={d:copy.deepcopy(settings) for d in DATASETS}
# The SHD R0 backend is distinct from the existing Spike/Texture bundles.
METHODS['gatezo']['shd']['native_args'] = json.loads((Path(__file__).parent /
    'gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json').read_text(encoding='utf-8'))[0]
for dataset in ('spike', 'texture'):
    METHODS['gatezo'][dataset] = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, hidden_size=200, layers=1,
        pool_size=100, growth_step=3, zo_min_keep=1, zo_sparsity_lambda=1e-4,
        zo_sample_size=0, ridge=0., validation_ratio=.2, calibration_ratio=.5,
        growth_failure_patience=10, growth_min_delta=1e-4)
METHODS['efficienttrain'] = {d: dict(epochs=EPOCHS, batch_size=BATCH_SIZE, hidden_size=0, layers=0,
    lr=.004, input_size=224, warmup_epochs=20, mixup=0., cutmix=0.,
    smoothing=.1, weight_decay=.05, model='resnet50',
    protocol='efficienttrain_pp_resnet50_v1', image_encoding='train_range_uint8') for d in DATASETS}
SEEDS = [57, 58, 59, 60, 61]

# SHD native learning profiles. Data representation, seeds, epochs and batch
# size remain PP3-controlled. Do not apply these overrides to other datasets.
# Sources: SPSN run.py/network.py (SB basic); TP experiments/shd.sh (FF TP);
# SNN_FF FF_SNN/SNNFF_SHD.py; ESPP main.py SHD branch + analyze_shd.ipynb
# final non-augmented, non-concatenated readout (not checkpoint-sweep settings).
SHD_NATIVE_PROFILES = {
    'spsn': dict(hidden_size=128, layers=3, lr=.001, tau_mem=.02, tau_syn=.02, time_step=.001,
                 spike_mode='SB', loss_mode='mean', reg_thr=0.),
    'tp': dict(hidden_size=450, layers=1, lr=.0001, layerwise=True,
               l_vth=1., l_leak_m=.96, l_leak_t=.97, l_rst_type='soft',
               l_out_leak_m=1., l_rec=False, norm_type='none', surrogate_type='1',
               surrogate_scale=1., train_s=False, optim='Adam',
               scheduler_name='CosineAnnealingLR', T=1),
    'ffsnn': dict(hidden_size=500, layers=2, lr=.001, beta=.9,
                  threshold=5., loss_alpha=4.),
    'espp': dict(hidden_size=200, layers=3, lr=.0001, beta=[.94,.96,.98,1.],
                 online=True, recurrency_type='dense', c_y=[1.5,-1.5], inp_thr=.05,
                 readout_lr=.0003, readout_weight_decay=0., readout_cat=False),
}
for method, profile in SHD_NATIVE_PROFILES.items():
    METHODS[method]['shd'].update(copy.deepcopy(profile))
# Preserve the native relation: model integration step equals event-bin width.
RUNTIME = dict(workers=0, threads=2, seed=57, output='/home/yizhil/pp3/result', power_sample_seconds=.1)
# Smoke runs intentionally keep their small execution budget.
SMOKE = dict(samples=4, steps=16, hidden_size=8, layers=1, batch_size=3, epochs=1)
VALIDATION = dict(ratio=.2, speakers=2)


def make_config(method, dataset):
    config = dict(method=method, dataset=dataset, data=copy.deepcopy(DATASETS[dataset]),
                model=copy.deepcopy(METHODS[method][dataset]), validation=copy.deepcopy(VALIDATION),
                **copy.deepcopy(RUNTIME))
    if dataset == 'shd' and method in ('espp', 'ffsnn'):
        # Native sequence lengths; preserve PP3 event augmentation, binary
        # occupancy and the full one-second window. Rebin, never subsample.
        steps = 100 if method == 'espp' else 10
        config['data'].update(steps=steps, dt=config['data']['max_time']/steps)
        config['data']['binning_protocol'] = 'native_bin_count_pp3_fixed_window_binary_v1'
    if dataset == 'shd' and method == 'spsn':
        config['model']['time_step'] = config['data']['dt']
    if dataset == 'shd' and method != 'gatezo':
        config['hyperparameter_protocol'] = ('native_shd_learning_pp3_data_v3'
            if method in SHD_NATIVE_PROFILES else 'pp3_adapter_no_native_shd_profile_v1')
        if method in ('espp', 'ffsnn'):
            config['hyperparameter_protocol'] = 'native_shd_learning_native_bin_count_pp3_augmentation_v4'
    return config
