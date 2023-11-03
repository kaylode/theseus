import pandas as pd
from tqdm import tqdm

from theseus.base.utilities.loggers import LoggerObserver

tqdm.pandas()
from .name_filter import FilterColumnNames

LOGGER = LoggerObserver.getLogger("main")

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    use_parallel = True
except:
    use_parallel = False
    LOGGER.text(
        "pandarallel should be installed for parallerization. Using normal apply-function instead",
        level=LoggerObserver.WARN,
    )


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

    def apply(self, df, function, parallel=True, axis=0, show_progress=True):

        df_func = df.apply
        if use_parallel and parallel:
            if not isinstance(df, pd.core.groupby.SeriesGroupBy):
                df_func = df.parallel_apply
        else:
            if show_progress:
                df_func = df.progress_apply

        if isinstance(df, pd.DataFrame):
            kwargs = {"axis": axis}
        else:
            kwargs = {}

        return df_func(function, **kwargs)

    def prerun(self, df):
        if self.filter is not None:
            self.column_names = self.filter.run(df)

    def run(self, df):
        return df

    def log(self, text, level=LoggerObserver.INFO):
        if self.verbose:
            LOGGER.text(text, level=level)
