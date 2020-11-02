import torch
import torch.nn as nn
import numpy as np
from .metrics.metrictemplate import TemplateMetric

def get_distance_fn(distance):
    if distance == 'cosine':
        return cosine_distance
    if distance == 'euclide':
        return euclide_distance

def euclide_distance(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()

    return dist_mtx

def cosine_distance(emb1, emb2):
    distance = nn.CosineSimilarity()
    return distance(emb1, emb2)

class Similarity(TemplateMetric):
    def __init__(self, distance='cosine', decimals = 10):
        self.reset()
        self.decimals = decimals
        self.distance_name = distance
        self.distance = get_distance_fn(distance)
        
    def compute(self, output, target):
        return self.distance(output, target)
        

    def update(self,  output, target):
        batch_size = target.shape[0]
        score = self.compute(output, target)
        self.total += score.sum()
        self.sample_size += batch_size
        
    def reset(self):
        self.sample_size = 0
        self.total = 0
        
    def value(self):
        values = self.total * 1.0 / self.sample_size
        if values.is_cuda:
            values = values.cpu()
        return {f"{self.distance_name}_distance" : np.around(values.item(), decimals = self.decimals)}

    def __str__(self):
        return f'Similarity: {self.value()}'

    def __len__(self):
        return len(self.sample_size)