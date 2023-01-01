from .coco import COCODataset
from theseus.base.datasets import DATASET_REGISTRY

DATASET_REGISTRY.register(COCODataset)