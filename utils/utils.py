import torch
import torch.nn as nn
import torchvision
import numpy as np
import math

def one_hot_embedding(labels, num_classes):
    '''
    Embedding labels to one-hot form.

    :param labels: (LongTensor) class labels, sized [N,].
    :param num_classes: (int) number of classes.
    :return: (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]
