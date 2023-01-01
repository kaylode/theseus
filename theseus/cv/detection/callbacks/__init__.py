from .visualization import DetectionVisualizerCallbacks
from theseus.base.callbacks import CALLBACKS_REGISTRY

CALLBACKS_REGISTRY.register(DetectionVisualizerCallbacks)