from theseus.base.callbacks import CALLBACKS_REGISTRY

from .visualization import DetectionVisualizerCallback

CALLBACKS_REGISTRY.register(DetectionVisualizerCallback)
