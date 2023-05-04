from theseus.registry import Registry

from .metric_template import Metric

METRIC_REGISTRY = Registry("METRIC")

from .accuracy import *
from .bl_accuracy import *
from .confusion_matrix import *
from .f1 import *
from .mcc import *
from .precision_recall import *
from .roc_auc_score import *

METRIC_REGISTRY.register(Accuracy)
METRIC_REGISTRY.register(BalancedAccuracyMetric)
METRIC_REGISTRY.register(F1ScoreMetric)
METRIC_REGISTRY.register(ConfusionMatrix)
METRIC_REGISTRY.register(PrecisionRecall)
METRIC_REGISTRY.register(ROCAUCScore)
METRIC_REGISTRY.register(MCC)
