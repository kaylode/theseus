from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class DropColumns(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        self.prerun(df)
        df = df.drop(self.column_names, axis=1)
        self.log(f"Dropped columns: {self.column_names}")
        return df


class LambdaDropRows(Preprocessor):
    def __init__(self, lambda_func, **kwargs):
        super().__init__(**kwargs)
        self.lambda_func = lambda_func

    def run(self, df):
        self.prerun(df)
        ori_size = df.shape[0]
        df = df.drop(df[self.apply(df, self.lambda_func, parallel=True, axis=1)].index)
        dropped_size = ori_size - df.shape[0]
        self.log(f"Dropped {dropped_size} rows based on lambda function")
        return df


class DropDuplicatedRows(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        num_duplicates = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        self.log(f"Dropped {num_duplicates} duplicated rows")
        return df


class DropEmptyColumns(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        cols_to_use = [
            idx for idx, val in (df.isna().mean() >= 1.0).items() if val == False
        ]
        empty_cols = set(df.columns) - set(cols_to_use)
        df = df.loc[:, cols_to_use]
        self.log(f"Dropped empty columns: {empty_cols}")
        return df


class DropSingleValuedColumns(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        single_val_cols = [
            idx
            for idx, val in df.nunique().items()
            if val <= 1 and df[idx].dtype not in [int, float]
        ]
        df = df.drop(single_val_cols, axis=1, inplace=False)
        self.log(f"Dropped single-valued columns: {single_val_cols}")
        return df
