from .base import Preprocessor
from theseus.utilities.loggers.observer import LoggerObserver

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
            self.log('Column names not specified. Automatically categorizing columns with non-defined types', level=LoggerObserver.WARN)
            self.column_names = [ col  for col, dt in df.dtypes.items() if dt == object]
            for column_name in self.column_names:
                df[column_name] = df[column_name].astype("category")
        self.log(f'Categorized columns: {self.column_names}')
        return df