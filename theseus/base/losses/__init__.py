from theseus.registry import Registry

LOSS_REGISTRY = Registry('LOSS')

from .multi_loss import MultiLoss
from .ce_loss import *
from .focal_loss import FocalLoss
from .mse_loss import MeanSquaredErrorLoss

LOSS_REGISTRY.register(MultiLoss)
LOSS_REGISTRY.register(ClassificationCELoss)
LOSS_REGISTRY.register(FocalLoss)
LOSS_REGISTRY.register(MeanSquaredErrorLoss)