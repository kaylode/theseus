from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class LambdaCreateColumn(Preprocessor):
    def __init__(self, target_column, lambda_func, **kwargs):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.lambda_func = lambda_func

    def run(self, df):
        self.prerun(df)

        lambda_dict = {self.target_column: self.lambda_func}

        df = df.assign(**lambda_dict)
        self.log(f"Created new columns: {self.target_column}")
        return df
