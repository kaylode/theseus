from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .csv_dataset import *
from .folder_dataset import *

DATASET_REGISTRY.register(ClassificationCSVDataset)
DATASET_REGISTRY.register(ClassificationImageFolderDataset)

from .mixupcutmix_collator import MixupCutmixCollator

DATALOADER_REGISTRY.register(MixupCutmixCollator)
