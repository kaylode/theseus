from torckay.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .dataset import *
from .dataloader import *

DATASET_REGISTRY.register(CSVDataset)
DATALOADER_REGISTRY.register(CSVLoader)