from torckay.registry import Registry

TRANSFORM_REGISTRY = Registry('TRANSFORM')

from . import albumentation, torchvision