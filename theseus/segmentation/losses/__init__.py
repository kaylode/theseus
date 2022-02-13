from theseus.base.losses import LOSS_REGISTRY

from .ce_loss import *
from .dice_loss import DiceLoss, BinaryDiceLoss

LOSS_REGISTRY.register(CELoss)
LOSS_REGISTRY.register(SmoothCELoss)
LOSS_REGISTRY.register(DiceLoss)
LOSS_REGISTRY.register(BinaryDiceLoss)