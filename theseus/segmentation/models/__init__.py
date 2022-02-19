from theseus.base.models import MODEL_REGISTRY

from .unet import UNetWrapper
from .deeplab import DeepLabV3
from .segformer import SegFormer
from .new_unet import BaseSegModel
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(UNetWrapper)
MODEL_REGISTRY.register(DeepLabV3)
MODEL_REGISTRY.register(SegFormer)
MODEL_REGISTRY.register(BaseSegModel)