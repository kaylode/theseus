from torckay.base.losses import LOSS_REGISTRY

from .ce_loss import *

LOSS_REGISTRY.register(CELoss)
LOSS_REGISTRY.register(SmoothCELoss)