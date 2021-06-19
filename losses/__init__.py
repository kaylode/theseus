from .focalloss import FocalLoss
from .smoothceloss import smoothCELoss
import torch.nn as nn

def get_loss(name):
    if name == 'focal':
        return FocalLoss()
    if name == 'smoothce':
        return smoothCELoss()
    if name == 'ce':
        return nn.CrossEntropyLoss()