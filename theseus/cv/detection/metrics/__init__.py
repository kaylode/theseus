from theseus.base.metrics import METRIC_REGISTRY

from .map import MeanAveragePrecision
from .precision_recall import DetectionPrecisionRecall

METRIC_REGISTRY.register(MeanAveragePrecision)
METRIC_REGISTRY.register(DetectionPrecisionRecall)
