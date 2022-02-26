from theseus.base.metrics import METRIC_REGISTRY

from .dicecoeff import *
from .pixel_accuracy import *
from .miou import *

METRIC_REGISTRY.register(PixelAccuracy)
METRIC_REGISTRY.register(mIOU)
METRIC_REGISTRY.register(DiceScore)