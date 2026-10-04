import os
import random

import numpy as np
import torch


def set_global_seed(default_seed=None):
    raw_seed = os.environ.get("BDNN_SEED")
    if raw_seed in ("", "none", "None", "random", "Random"):
        return None
    if raw_seed is None and default_seed is None:
        return None
    seed = int(raw_seed if raw_seed is not None else default_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
