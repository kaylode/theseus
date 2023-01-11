from theseus.base.metrics import METRIC_REGISTRY

from .errorcases import *
from .projection import *

METRIC_REGISTRY.register(ErrorCases)
METRIC_REGISTRY.register(EmbeddingProjection)
