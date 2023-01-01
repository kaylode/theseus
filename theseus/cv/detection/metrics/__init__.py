from .map import MeanAveragePrecision
from .precision_recall import DetectionPrecisionRecall

from theseus.base.metrics import METRIC_REGISTRY
METRIC_REGISTRY.register(MeanAveragePrecision)
METRIC_REGISTRY.register(DetectionPrecisionRecall)