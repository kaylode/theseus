from metrics import *
from datasets import *
from losses import *
from models import *
from trainer import *
from augmentations import *
from loggers import *
from configs import *


import torch
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from .random_seed import seed_everything


def get_instance(config, **kwargs):
    # Inherited from https://github.com/vltanh/pytorch-template
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)