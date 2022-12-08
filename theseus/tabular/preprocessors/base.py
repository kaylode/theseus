from theseus.utilities.loggers import LoggerObserver
from .name_filter import FilterColumnNames
LOGGER = LoggerObserver.getLogger("main")

class Preprocessor:
    def __init__(self, column_names=None, verbose=False, **kwargs):
        self.verbose = verbose
        self.column_names = column_names

        self.filter = None
        if column_names is not None:
            self.filter = FilterColumnNames(patterns=column_names)

    def prerun(self, df):
        if self.filter is not None:
            self.column_names = self.filter.run(df)

    def run(self, df):
        return df

    def log(self, text, level=LoggerObserver.INFO):
        if self.verbose:
            LOGGER.text(text, level=level)