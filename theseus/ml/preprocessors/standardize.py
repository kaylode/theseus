import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler

from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class Standardize(Preprocessor):
    def __init__(self, method="normalizer", **kwargs):
        super().__init__(**kwargs)
        self.method = method
        if method == "normalizer":
            self.func = Normalizer()
        elif method == "robust":
            self.func = RobustScaler()
        elif method == "minmax":
            self.func = MinMaxScaler()
        elif method == "standard":
            self.func = StandardScaler()
        else:
            self.func = None

    def run(self, df):
        self.prerun(df)
        if self.column_names is not None:
            for column_name in self.column_names:
                df[column_name] = self.func.fit_transform(pd.DataFrame(df[column_name]))
        else:
            self.log(
                "Column names not specified. Standardize all columns",
                level=LoggerObserver.ERROR,
            )
            self.column_names = list(df.columns)
            df = self.func.fit_transform(df)
        self.log(f"Standardized columns with {self.method}: {self.column_names}")
        return df
