from .base import Preprocessor
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class Aggregate(Preprocessor):
    def __init__(self, aggregation_list, **kwargs):
        super().__init__(**kwargs)

        # aggregation_list should be [{target_name: str, aggr_method: [str, def], based_columns=[cols]}]
        self.aggregation_list = aggregation_list

    def run(self, df):
        self.prerun(df)

        new_column_names = []
        for item in self.aggregation_list:
            method_name = item['aggr_method']
            target_name = item['target_name']
            based_columns = item['based_columns']

            if isinstance(method_name, str):
                if method_name == 'sum':
                    df[target_name] = df[based_columns].sum(axis=1)
                if method_name == 'mean':
                    df[target_name] = df[based_columns].mean(axis=1)
                if method_name == 'subtract':
                    df[target_name] = df[based_columns].sub(axis=1)

            elif callable(method_name):
                df[target_name] = self.apply(df[based_columns], function=method_name, axis=1)
            else:
                LOGGER.text('Unsuppported aggregation method', level=LoggerObserver.ERROR)
                raise ValueError()
            new_column_names.append(target_name)

        self.log(f'Aggregated new columns: {new_column_names}')
        return df