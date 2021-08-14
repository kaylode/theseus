import torch.nn as nn

class FeatureProjection(nn.Module):
    """
    Projects instance features into a space of dimensionality 
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

class SpatialEncoding(nn.Module):
    """
    Encodes bounding box coordinates and relative sizes as vector of dimensionality 
    """
    def __init__(self, output_dim):
        super().__init__()
        self.linear = nn.Linear(5, output_dim)

    def forward(self, x):
        return self.linear(x)

class MergedProjection(nn.Module):
    """
    Merge instance projection with spatial encoding
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.projection = FeatureProjection(input_dim, output_dim)
        self.spatial = SpatialEncoding(output_dim)
    def forward(self, feats, boxes):
        feats = self.projection(feats)
        spatial_feats = self.spatial(boxes)
        outputs = feats + spatial_feats
        return self.dropout(outputs)
