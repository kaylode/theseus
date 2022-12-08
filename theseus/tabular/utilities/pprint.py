import pandas as pd
from tabulate import tabulate
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

def pretty_print_df(df, showindex=False):
    if isinstance(df, pd.DataFrame):
        LOGGER.text('\n'+
            tabulate(df, headers=df.columns, tablefmt='psql', showindex=showindex),
            level=LoggerObserver.INFO
        )
    elif isinstance(df, pd.Series):
        df_list = [i for i in zip(df.index.values.tolist(), df.values.tolist())]
        LOGGER.text('\n'+
            tabulate(df_list, headers='keys', tablefmt='psql', showindex=showindex),
            level=LoggerObserver.INFO
        )
    else:
        raise ValueError()