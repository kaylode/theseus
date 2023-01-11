from theseus.base.callbacks import CALLBACKS_REGISTRY

from .visualization import DetectionVisualizerCallbacks

CALLBACKS_REGISTRY.register(DetectionVisualizerCallbacks)
