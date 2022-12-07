import pandas as pd
from .base import Preprocessor
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class ToDatetime(Preprocessor):
    def __init__(self, column_names, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)
        self.column_names = column_names

    def run(self, df):
        for column_name in self.column_names:
            df[column_name] = pd.to_datetime(df[column_name])

        self.log(f'Converted to datetime: {self.column_names}')
        return df

class DateDecompose(Preprocessor):
    def __init__(self, column_names, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)
        self.column_names = column_names

    def run(self, df):
        for column_name in self.column_names:
            df[column_name+'_day'] = pd.to_datetime(df[column_name]).dt.day
            df[column_name+'_month'] = pd.to_datetime(df[column_name]).dt.month
            df[column_name+'_year'] = pd.to_datetime(df[column_name]).dt.year
            df.drop(columns=column_name, inplace=True)

        self.log(f'Decomposed to datetime: {self.column_names}')
        return df