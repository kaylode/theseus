from theseus.base.models import MODEL_REGISTRY

from .segformer import SegFormer
from .segmodels import BaseSegModel
from .lawin import Lawin
from .hardnet import FCHarDNet
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(Lawin)
MODEL_REGISTRY.register(SegFormer)
MODEL_REGISTRY.register(FCHarDNet)
MODEL_REGISTRY.register(BaseSegModel)