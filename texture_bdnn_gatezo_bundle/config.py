# config.py
SCN_DEFAULTS = {
    'seed': 2027,
    'L_max': 200,
    'T_max': 100,
    'tol': 0.0005,
    'Lambdas': [0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200],
    'r': [0.99, 0.999, 0.9999, 0.99999, 0.999999],
    'nB': 3,
    'verbose': 7,
    'zo_steps': 0,
    'zo_lr': 0.001,
    'zo_mu': 0.01,
    'zo_alpha': 0.0,
    'zo_directions': 1,
    'zo_direction_mode': 'random', # random or residual
    'zo_residual_top_ratio': 0.01,
    'zo_residual_mix': 0.2,
    'zo_converge': False,
    'zo_batch_size': 128,
    'zo_max_epochs': 1,
    'zo_accept_tol': 0.0,
    'zo_optimize_u': False,
    'zo_sample_size': None,
    'zo_verbose': False
}
