import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


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
            errorbar=None,
            color="#69b3a2",
            ax=ax,
        )

        ax2.set(xlabel="class id", ylabel="number of instances")

    def sample_dimension_dist(self, ax=None):
        # plot a bar chart
        ax = sns.scatterplot(x="width", y="height", data=self.sample_df, ax=ax)

    def instance_dimension_dist(self, ax=None):
        # plot a bar chart
        sns.scatterplot(x="width", y="height", data=self.instance_df, ax=ax)

    def save(self, path):
        plt.savefig(path)
