from torckay.registry import Registry
from .custom import Denormalize

TRANSFORM_REGISTRY = Registry('TRANSFORM')
from . import albumentation, torchvision
TRANSFORM_REGISTRY.register(Denormalize, prefix='Custom')