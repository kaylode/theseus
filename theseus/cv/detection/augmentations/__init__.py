from theseus.cv.base.augmentations import TRANSFORM_REGISTRY
from albumentations import BboxParams
from .bbox_transforms import BoxOrder, BoxNormalize
from .compose import DetCompose

TRANSFORM_REGISTRY.register(BboxParams, prefix='Alb')
TRANSFORM_REGISTRY.register(BoxOrder)
TRANSFORM_REGISTRY.register(BoxNormalize)
TRANSFORM_REGISTRY.register(DetCompose)
