import os
import random

import numpy as np
import torch

SEED = 1702


def seed_everything(seed: int = SEED):
    """
    https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
