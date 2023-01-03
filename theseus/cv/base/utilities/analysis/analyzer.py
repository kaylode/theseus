import matplotlib.pyplot as plt
import numpy as np

from theseus.base.utilities.analyzer import Analyzer


class ClassificationAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

    def init_dict(self):
        self.instance_dict["id"] = []
        self.instance_dict["image_id"] = []
        self.instance_dict["class"] = []

        self.sample_dict["id"] = []
        self.sample_dict["width"] = []
        self.sample_dict["height"] = []

    def update_item(self, item):
        label = item["target"]["labels"][0]
        width, height = item["ori_size"]

        self.instance_dict["id"].append(len(self.instance_dict["id"]) + 1)
        self.instance_dict["image_id"].append(len(self.sample_dict["id"]) + 1)
        self.instance_dict["class"].append(label)

        self.sample_dict["id"].append(len(self.sample_dict["id"]) + 1)
        self.sample_dict["width"].append(width)
        self.sample_dict["height"].append(height)

    def analyze(self, figsize=(8, 8)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        self.sample_dimension_dist(axs[0])
        self.class_dist(axs[1])
        return fig


class SemanticAnalyzer(ClassificationAnalyzer):
    def __init__(self):
        super().__init__()

    def update_item(self, item):
        mask = item["target"]["mask"]
        width, height = item["ori_size"]

        unique_ids = np.unique(mask.numpy())

        for label in unique_ids:
            self.instance_dict["id"].append(len(self.instance_dict["id"]) + 1)
            self.instance_dict["image_id"].append(len(self.sample_dict["id"]) + 1)
            self.instance_dict["class"].append(int(label))

        self.sample_dict["id"].append(len(self.sample_dict["id"]) + 1)
        self.sample_dict["width"].append(width)
        self.sample_dict["height"].append(height)
