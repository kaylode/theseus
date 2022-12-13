from .base import Preprocessor
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class DropColumns(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        self.prerun(df)
        df = df.drop(self.column_names, axis=1)
        self.log(f'Dropped columns: {self.column_names}')
        return df

class DropDuplicatedRows(Preprocessor):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def run(self, df):
        num_duplicates = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        self.log(f'Dropped {num_duplicates} duplicated rows')
        return df
        
class DropEmptyColumns(Preprocessor):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
    
    def run(self, df):
        cols_to_use = [idx for idx,val in (df.isna().mean()>=1.).items() if val==False]
        empty_cols = set(df.columns)-set(cols_to_use)
        df = df.loc[:, cols_to_use]
        self.log(f'Dropped empty columns: {empty_cols}')
        return df

class DropSingleValuedColumns(Preprocessor):
    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def run(self, df):
        single_val_cols = [idx for idx,val in df.nunique().items() if val <=1 and df[idx].dtype not in [int, float] ]
        df = df.drop(single_val_cols, axis=1, inplace=False)
        self.log(f'Dropped single-valued columns: {single_val_cols}')
        return df