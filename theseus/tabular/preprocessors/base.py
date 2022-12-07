from theseus.utilities.loggers import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class Preprocessor:
    def __init__(self, verbose, **kwargs):
        self.verbose = verbose

    def run(self, df):
        return df

    def log(self, text, level=LoggerObserver.INFO):
        if self.verbose:
            LOGGER.text(text, level=level)