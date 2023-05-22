from theseus.base.metrics import METRIC_REGISTRY

from .acccuracy import SKLAccuracy, SKLBalancedAccuracyMetric
from .f1_score import SKLF1ScoreMetric
from .precision_recall import SKLPrecisionRecall
from .projection import SKLEmbeddingProjection
from .mcc import SKLMCC
from .roc_auc_score import SKLROCAUCScore

METRIC_REGISTRY.register(SKLPrecisionRecall)
METRIC_REGISTRY.register(SKLF1ScoreMetric)
METRIC_REGISTRY.register(SKLAccuracy)
METRIC_REGISTRY.register(SKLBalancedAccuracyMetric)
METRIC_REGISTRY.register(SKLEmbeddingProjection)
METRIC_REGISTRY.register(SKLMCC)
METRIC_REGISTRY.register(SKLROCAUCScore)
