import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Analyzer:
    def __init__(self):
        self.instance_dict = {}
        self.sample_dict = {}
        self.init_dict()

    def init_dict(self):
        raise NotImplementedError

    def update_item(self, item):
        raise NotImplementedError

    def analyze(self):
        raise NotImplementedError

    def add_dataset(self, dataset):
        self.class_names = dataset.classnames
        for i, item in enumerate(tqdm(dataset)):
            self.update_item(item)

        self.instance_df = pd.DataFrame.from_dict(self.instance_dict)
        self.sample_df = pd.DataFrame.from_dict(self.sample_dict)

    def class_dist(self, ax=None):
        # plot a bar chart
        ax2 = sns.barplot(
            x="class", 
            y="id", 
            data=self.instance_df, 
            estimator=len, 
            ci=None, 
            color='#69b3a2',
            ax=ax)

        ax2.set(xlabel='class id', ylabel='number of instances')

    def sample_dimension_dist(self, ax=None):
        # plot a bar chart
        ax = sns.scatterplot(
            x="width", 
            y="height", 
            data=self.sample_df,
            ax=ax)

    def instance_dimension_dist(self, ax=None):
        # plot a bar chart
        sns.scatterplot(
            x="width", 
            y="height", 
            data=self.instance_df,
            ax=ax)

    def save(self, path):
        plt.savefig(path)

class ClassificationAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

    def init_dict(self):
        self.instance_dict['id'] = []
        self.instance_dict['image_id'] = []
        self.instance_dict['class'] = []

        self.sample_dict['id'] = []
        self.sample_dict['width'] = []
        self.sample_dict['height'] = []

    def update_item(self, item):
        label = item['target']['labels'][0]
        width, height = item['ori_size']
        
        self.instance_dict['id'].append(len(self.instance_dict['id'])+1)
        self.instance_dict['image_id'].append(len(self.sample_dict['id'])+1)
        self.instance_dict['class'].append(label)

        self.sample_dict['id'].append(len(self.sample_dict['id'])+1)
        self.sample_dict['width'].append(width)
        self.sample_dict['height'].append(height)

    def analyze(self, figsize=(8,8)):
        fig, axs = plt.subplots(1, 2 ,figsize=figsize)

        self.sample_dimension_dist(axs[0])
        self.class_dist(axs[1])
        return fig

class SemanticAnalyzer(ClassificationAnalyzer):
    def __init__(self):
        super().__init__()

    def update_item(self, item):
        mask = item['target']['mask']
        width, height = item['ori_size']
        
        unique_ids = np.unique(mask.numpy())

        for label in unique_ids:
            self.instance_dict['id'].append(len(self.instance_dict['id'])+1)
            self.instance_dict['image_id'].append(len(self.sample_dict['id'])+1)
            self.instance_dict['class'].append(int(label))

        self.sample_dict['id'].append(len(self.sample_dict['id'])+1)
        self.sample_dict['width'].append(width)
        self.sample_dict['height'].append(height)