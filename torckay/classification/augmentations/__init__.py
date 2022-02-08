from torckay.base.augmentations import TRANSFORM_REGISTRY
from .custom import *

TRANSFORM_REGISTRY.register(CustomCutout, prefix='Custom')
TRANSFORM_REGISTRY.register(RandomCutmix, prefix='Custom')
TRANSFORM_REGISTRY.register(RandomMixup, prefix='Custom')
TRANSFORM_REGISTRY.register(Denormalize, prefix='Custom')