from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class MapValue(Preprocessor):
    """
    mapping_dict should be dict of dicts; each inside dict is a mapping dict
    """

    def __init__(self, mapping_dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mapping_dict = mapping_dict

    def run(self, df):
        self.prerun(df)

        for column_name in self.column_names:
            mapping_dict = self.mapping_dict[column_name]
            df[column_name] = df[column_name].map(mapping_dict)

        self.log(f"Column values changed based on mapping: {self.column_names}")
        return df
