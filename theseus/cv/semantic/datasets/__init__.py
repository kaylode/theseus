from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .csv_dataset import SemanticCSVDataset

DATASET_REGISTRY.register(SemanticCSVDataset)

from .mosaic_collator import MosaicCollator

DATALOADER_REGISTRY.register(MosaicCollator)
