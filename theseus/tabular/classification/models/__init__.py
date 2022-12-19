from theseus.base.models import MODEL_REGISTRY

from .gbms import GBClassifiers
MODEL_REGISTRY.register(GBClassifiers)