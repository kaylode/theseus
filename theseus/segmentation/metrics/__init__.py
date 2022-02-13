from theseus.base.metrics import METRIC_REGISTRY

from .dicecoeff import *
from .pixel_accuracy import *


METRIC_REGISTRY.register(PixelAccuracy)
METRIC_REGISTRY.register(DiceScore)