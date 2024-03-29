from theseus.base.losses import LOSS_REGISTRY

from .ce_loss import *
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszSoftmax
from .tversky_loss import FocalTverskyLoss

LOSS_REGISTRY.register(SemanticCELoss)
LOSS_REGISTRY.register(OhemCELoss)
LOSS_REGISTRY.register(SemanticSmoothCELoss)
LOSS_REGISTRY.register(DiceLoss)
LOSS_REGISTRY.register(FocalTverskyLoss)
LOSS_REGISTRY.register(LovaszSoftmax)
