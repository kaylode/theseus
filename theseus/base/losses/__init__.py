from theseus.registry import Registry

LOSS_REGISTRY = Registry('LOSS')

from .multi_loss import MultiLoss

LOSS_REGISTRY.register(MultiLoss)