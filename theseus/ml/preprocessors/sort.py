from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class SortBy(Preprocessor):
    def __init__(self, ascending=True, **kwargs):
        super().__init__(**kwargs)
        self.ascending = ascending

    def run(self, df):
        self.prerun(df)

        self.log(f"Sort rows by: {self.column_names}")
        return df.sort_values(by=self.column_names, ascending=self.ascending)
