from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .coco import COCODataset

DATASET_REGISTRY.register(COCODataset)
