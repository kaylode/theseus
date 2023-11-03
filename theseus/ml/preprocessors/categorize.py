import pandas as pd

from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class Categorize(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        self.prerun(df)
        if self.column_names is not None:
            for column_name in self.column_names:
                df[column_name] = df[column_name].astype("category")
        else:
            self.log(
                "Column names not specified. Automatically categorizing columns with non-defined types",
                level=LoggerObserver.WARN,
            )
            self.column_names = [col for col, dt in df.dtypes.items() if dt == object]
            for column_name in self.column_names:
                df[column_name] = df[column_name].astype("category")
        self.log(f"Categorized columns: {self.column_names}")
        return df


class EnforceType(Preprocessor):
    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        assert type in [
            "str",
            "int",
            "float",
            "datetime",
            "category",
            "bool",
        ], "Unsupported type enforcing"

    def run(self, df):
        self.prerun(df)
        if self.column_names is None:
            self.column_names = [col for col, dt in df.dtypes.items() if dt == object]

            self.log(
                "Column names not specified. Automatically categorizing columns with non-defined types",
                level=LoggerObserver.WARN,
            )

        for column_name in self.column_names:
            if type == "str":
                df[column_name] = df[column_name].astype(str)
            elif type == "int":
                df[column_name] = df[column_name].astype(int)
            elif type == "category":
                df[column_name] = df[column_name].astype("category")
            elif type == "float":
                df[column_name] = df[column_name].astype(float)
            elif type == "datetime":
                df[column_name] = pd.to_datetime(df[column_name])
            elif type == "bool":
                df[column_name] = df[column_name].astype(bool)
            else:
                df[column_name] = df[column_name].astype(object)

        self.log(f"{self.type}-enforced columns: {self.column_names}")
        return df
