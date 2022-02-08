from torch.optim import SGD, Adam

from torckay.registry import Registry

from . import lr_scheduler, scalers

OPTIM_REGISTRY = Registry('OPTIMIZER')
OPTIM_REGISTRY.register(Adam)
OPTIM_REGISTRY.register(SGD)
