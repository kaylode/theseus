from typing import List

class ChainCollateWrapper(object):
    """Wrapper for list of collate functions
    
    """
    def __init__(self, pre_collate_fns: List, **kwargs):
        self.pre_collate_fns = pre_collate_fns

    def __call__(self, batch):
        for fn in self.pre_collate_fns:
            batch = fn(batch)
        return batch