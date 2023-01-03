from theseus.registry import Registry

from .metric_template import Metric

METRIC_REGISTRY = Registry("METRIC")

from .accuracy import *
from .bl_accuracy import *
from .confusion_matrix import *
from .f1 import *

METRIC_REGISTRY.register(Accuracy)
METRIC_REGISTRY.register(BalancedAccuracyMetric)
METRIC_REGISTRY.register(F1ScoreMetric)
METRIC_REGISTRY.register(ConfusionMatrix)
