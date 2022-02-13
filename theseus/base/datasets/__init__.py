from theseus.registry import Registry
from torch.utils.data import DataLoader, Dataset

DATASET_REGISTRY = Registry('DATASET')
DATALOADER_REGISTRY = Registry('DATALOADER')

DATASET_REGISTRY.register(Dataset)
DATALOADER_REGISTRY.register(DataLoader)