from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY

from .csv_dataset import TabularCSVDataset

DATASET_REGISTRY.register(TabularCSVDataset)
