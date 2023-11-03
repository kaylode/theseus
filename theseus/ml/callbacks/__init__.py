from theseus.registry import Registry

from .base_callbacks import Callbacks, CallbacksList
from .checkpoint_callbacks import SKLearnCheckpointCallbacks
from .explainer import (
    LIMEExplainer,
    PartialDependencePlots,
    PermutationImportance,
    ShapValueExplainer,
)
from .metric_callbacks import MetricLoggerCallbacks
from .optuna_callbacks import OptunaCallbacks

CALLBACKS_REGISTRY = Registry("CALLBACKS")

CALLBACKS_REGISTRY.register(SKLearnCheckpointCallbacks)
CALLBACKS_REGISTRY.register(ShapValueExplainer)
CALLBACKS_REGISTRY.register(PermutationImportance)
CALLBACKS_REGISTRY.register(PartialDependencePlots)
CALLBACKS_REGISTRY.register(LIMEExplainer)
CALLBACKS_REGISTRY.register(OptunaCallbacks)
CALLBACKS_REGISTRY.register(MetricLoggerCallbacks)
