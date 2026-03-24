from theseus.registry import Registry

LOSS_REGISTRY = Registry("LOSS")

from .bce_loss import BCELoss
from .ce_loss import *
from .focal_loss import FocalLoss
from .mae_loss import MeanAbsoluteErrorLoss
from .mse_loss import MeanSquaredErrorLoss
from .multi_loss import MultiLoss

LOSS_REGISTRY.register(MultiLoss)
LOSS_REGISTRY.register(ClassificationCELoss)
LOSS_REGISTRY.register(FocalLoss)
LOSS_REGISTRY.register(MeanSquaredErrorLoss)
LOSS_REGISTRY.register(ClassificationSmoothCELoss)
LOSS_REGISTRY.register(BCELoss)
LOSS_REGISTRY.register(MeanAbsoluteErrorLoss)
