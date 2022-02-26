from torch.utils.data import DataLoader
from .collator import ChainCollateWrapper

class DataLoaderWithCollator(DataLoader):
    def __init__(self, dataset, collate_fn=None, sampler=None, **kwargs) -> None:
        self.dataset = dataset

        if collate_fn is not None:
            if isinstance(collate_fn, list):
                collate_fn.insert(0, dataset.collate_fn)
                collate_fn = ChainCollateWrapper(collate_fn)
            else:
                collate_fn = ChainCollateWrapper([dataset.collate_fn, collate_fn])
        else:
            collate_fn = dataset.collate_fn

        super().__init__(
          dataset=dataset, 
          collate_fn=collate_fn, 
          sampler=sampler,
          **kwargs)