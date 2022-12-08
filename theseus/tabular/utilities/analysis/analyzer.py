from theseus.tabular.utilities.pprint import pretty_print_df
from theseus.utilities.loggers import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

def check_nan(df):
    missing = df.isna().mean()*100
    return missing.sort_values(ascending=False, inplace=False)

def check_duplicate(df):
    num_duplicates = df.duplicated().sum()
    LOGGER.text(f'Number of duplicate rows: {num_duplicates}', level=LoggerObserver.INFO)

class DataFrameAnalyzer:
    def __init__(self, **kwargs):
        pass

    def analyze(self, df):
        missing_df = check_nan(df)
        pretty_print_df(missing_df)
        check_duplicate(df)
        