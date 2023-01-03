from torchvision.transforms import RandAugment  # torchvision 1.10
from torchvision.transforms.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAffine,
    RandomChoice,
    RandomErasing,
    RandomPerspective,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from . import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(RandomResizedCrop, prefix="TV")
TRANSFORM_REGISTRY.register(Normalize, prefix="TV")
TRANSFORM_REGISTRY.register(Resize, prefix="TV")
TRANSFORM_REGISTRY.register(Compose, prefix="TV")
TRANSFORM_REGISTRY.register(ToTensor, prefix="TV")
TRANSFORM_REGISTRY.register(RandAugment, prefix="TV")
TRANSFORM_REGISTRY.register(RandomAffine, prefix="TV")
TRANSFORM_REGISTRY.register(RandomPerspective, prefix="TV")
TRANSFORM_REGISTRY.register(RandomErasing, prefix="TV")
TRANSFORM_REGISTRY.register(RandomChoice, prefix="TV")
TRANSFORM_REGISTRY.register(ColorJitter, prefix="TV")
