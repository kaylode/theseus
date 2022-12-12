from theseus.base.models import MODEL_REGISTRY

from .neunets import NeuralNet
MODEL_REGISTRY.register(NeuralNet)