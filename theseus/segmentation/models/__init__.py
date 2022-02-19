from theseus.base.models import MODEL_REGISTRY

from .unet import UNetWrapper
from .deeplab import DeepLabV3
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(UNetWrapper)
MODEL_REGISTRY.register(DeepLabV3)