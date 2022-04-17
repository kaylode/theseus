from theseus.base.models import MODEL_REGISTRY

from .timm_models import *

MODEL_REGISTRY.register(BaseTimmModel)