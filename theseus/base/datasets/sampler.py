"""
Source: https://github.com/ufoym/imbalanced-dataset-sampler
"""

from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        op = getattr(dataset, '_calculate_classes_dist', None)
        if not callable(op):
            LOGGER.text("""Using BalanceSampler but _calculate_classes_dist()
            method is missing from the dataset""", LoggerObserver.ERROR)
            raise ValueError

        classes_dist = dataset._calculate_classes_dist()
        return classes_dist

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples