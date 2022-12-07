from theseus.cv.base.losses import LOSS_REGISTRY

from .ce_loss import *
from .focal_loss import FocalLoss

LOSS_REGISTRY.register(ClassificationCELoss)
LOSS_REGISTRY.register(ClassificationSmoothCELoss)
LOSS_REGISTRY.register(FocalLoss)