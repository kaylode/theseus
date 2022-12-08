from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from .base import Preprocessor
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class LabelEncode(Preprocessor):
    def __init__(self, encoder_type='le',  **kwargs):
        super().__init__(**kwargs)

        assert encoder_type in ['le', 'onehot', 'ordinal'], 'Encoder type not supported'

        self.encoder_type = encoder_type

        if self.encoder_type == 'le':
            self.encoder = LabelEncoder()
        elif self.encoder_type == 'onehot':
            self.encoder = OneHotEncoder()
        else:
            self.encoder = OrdinalEncoder()

    def run(self, df):
        self.prerun(df)
        if self.column_names is not None:
            for column_name in self.column_names:
                df[column_name] = self.encoder.fit_transform(df[column_name].values) 
        else:
            self.log('Column names not specified. Automatically label encode columns with non-defined types', level=LoggerObserver.WARN)
            column_names = [ col  for col, dt in df.dtypes.items() if dt == object]
            for column_name in column_names:
                df[column_name] = self.encoder.fit_transform(df[column_name].values) 

        self.log(f'Label-encoded columns: {column_name}')
        return df