from theseus.registry import Registry

from torch.optim import SGD, Adam, AdamW
from .schedulers import SchedulerWrapper


SCHEDULER_REGISTRY = Registry("SCHEDULER")
SCHEDULER_REGISTRY.register(SchedulerWrapper)

OPTIM_REGISTRY = Registry('OPTIMIZER')
OPTIM_REGISTRY.register(Adam)
OPTIM_REGISTRY.register(AdamW)
OPTIM_REGISTRY.register(SGD)
