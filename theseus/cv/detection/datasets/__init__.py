from .coco import COCODataset
from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

DATASET_REGISTRY.register(COCODataset)