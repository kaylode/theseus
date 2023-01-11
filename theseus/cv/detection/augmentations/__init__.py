from albumentations import BboxParams

from theseus.cv.base.augmentations import TRANSFORM_REGISTRY

from .bbox_transforms import BoxNormalize, BoxOrder
from .compose import DetCompose

TRANSFORM_REGISTRY.register(BboxParams, prefix="Alb")
TRANSFORM_REGISTRY.register(BoxOrder)
TRANSFORM_REGISTRY.register(BoxNormalize)
TRANSFORM_REGISTRY.register(DetCompose)
