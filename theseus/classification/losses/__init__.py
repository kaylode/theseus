from theseus.base.losses import LOSS_REGISTRY

from .ce_loss import *
from .focal_loss import FocalLoss

LOSS_REGISTRY.register(CELoss)
LOSS_REGISTRY.register(SmoothCELoss)
LOSS_REGISTRY.register(FocalLoss)