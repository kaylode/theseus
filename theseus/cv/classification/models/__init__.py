from theseus.cv.base.models import MODEL_REGISTRY

from .timm_models import *

MODEL_REGISTRY.register(BaseTimmModel)