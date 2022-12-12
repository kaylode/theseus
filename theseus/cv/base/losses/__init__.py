from theseus.base.losses import LOSS_REGISTRY

from .ce_loss import ClassificationSmoothCELoss
LOSS_REGISTRY.register(ClassificationSmoothCELoss)