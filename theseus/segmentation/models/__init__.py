from theseus.base.models import MODEL_REGISTRY

from .segformer import SegFormer
from .segmodels import BaseSegModel
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(SegFormer)
MODEL_REGISTRY.register(BaseSegModel)