import pandas as pd


class TabularCSVDataset:
    def __init__(
        self, data_path, target_column, txt_classnames=None, transform=None
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.target_column = target_column
        self.txt_classnames = txt_classnames

        if self.txt_classnames is not None:
            self.classnames = open(self.txt_classnames, "r").read().splitlines()
        else:
            self.classnames = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        if self.transform is not None:
            df = self.transform.run(df)
        (X, y) = (
            df.drop(self.target_column, axis=1).values,
            df[self.target_column].values,
        )

        return {
            "inputs": X,
            "targets": y,
            "feature_names": df.drop(self.target_column, axis=1).columns,
            "classnames": self.classnames,
            "target_name": self.target_column,
        }
