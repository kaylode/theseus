from theseus.cv.base.models import MODEL_REGISTRY

from .segmodels import BaseSegModel

MODEL_REGISTRY.register(BaseSegModel)
