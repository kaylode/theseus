from theseus.base.utilities.loggers import LoggerObserver

from .name_filter import FilterColumnNames

# try:
#     from pandarallel import pandarallel

#     pandarallel.initialize()
#     use_parallel = True
# except:
use_parallel = False

LOGGER = LoggerObserver.getLogger("main")


class Preprocessor:
    def __init__(
        self, column_names=None, exclude_columns=None, verbose=False, **kwargs
    ):
        self.verbose = verbose
        self.column_names = column_names

        self.filter = None
        if column_names is not None:
            self.filter = FilterColumnNames(
                patterns=column_names, excludes=exclude_columns
            )

    def apply(self, df, function, parallel=True, axis=0):
        if parallel:
            if not use_parallel:
                LOGGER.text(
                    "pandarallel should be installed for parallerization. Using normal apply-function instead",
                    level=LoggerObserver.WARN,
                )
                return df.apply(function, axis=axis)
            else:
                return df.parallel_apply(function, axis=axis)
        else:
            return df.apply(function, axis=axis)

    def prerun(self, df):
        if self.filter is not None:
            self.column_names = self.filter.run(df)

    def run(self, df):
        return df

    def log(self, text, level=LoggerObserver.INFO):
        if self.verbose:
            LOGGER.text(text, level=level)
