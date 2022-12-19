from theseus.base.metrics import METRIC_REGISTRY

from .precision_recall import SKLPrecisionRecall
from .f1_score import SKLF1ScoreMetric
from .acccuracy import SKLAccuracy, SKLBalancedAccuracyMetric

METRIC_REGISTRY.register(SKLPrecisionRecall)
METRIC_REGISTRY.register(SKLF1ScoreMetric)
METRIC_REGISTRY.register(SKLAccuracy)
METRIC_REGISTRY.register(SKLBalancedAccuracyMetric)
