from theseus.base.models import MODEL_REGISTRY

from .unet import UNetWrapper
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(UNetWrapper)