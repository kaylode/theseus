import numpy as np
import torch
import random

SEED = 1702

def seed_everything(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)