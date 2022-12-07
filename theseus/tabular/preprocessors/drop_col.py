from .base import Preprocessor
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class DropColumns(Preprocessor):
    def __init__(self, column_names, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)
        self.column_names = column_names

    def run(self, df):
        df = df.drop(self.column_names, axis=1)
        self.log(f'Dropped columns: {self.column_names}')
        return df

class DropColumnsWithNameFiltered(Preprocessor):
    def __init__(self, column_name_filtered, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)
        self.column_name_filtered = column_name_filtered

    def run(self, df):
        dropped_columns = []
        for filtered_col in self.column_name_filtered:
            filtered_l, filtered_r = None, None
            if filtered_col.startswith('*'):
                filtered_r = filtered_col.split('*')[1]
            elif filtered_col.endswith('*'):
                filtered_l = filtered_col.split('*')[0]
            elif '*' in filtered_col:
                filtered_l = filtered_col.split('*')[-1]
                filtered_r = filtered_col.split('*')[0]
            else:
                dropped_columns.append(filtered_col)
                df = df.drop(filtered_col, axis=1)
                continue

            for col_name in df.columns:
                if filtered_l and filtered_r:
                    if col_name.startswith(filtered_l) and col_name.endswith(filtered_r):
                        dropped_columns.append(col_name)
                        df = df.drop(col_name, axis=1)
                    continue	
                if filtered_l and col_name.startswith(filtered_l):
                    dropped_columns.append(col_name)
                    df = df.drop(col_name, axis=1)
                    continue
                if filtered_r and col_name.endswith(filtered_r):
                    dropped_columns.append(col_name)
                    df = df.drop(col_name, axis=1)
                    continue
                
        self.log(f'Dropped columns based on filter: {dropped_columns}')
        return df


class DropDuplicatedRows(Preprocessor):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)

    def run(self, df):
        num_duplicates = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        self.log(f'Dropped {num_duplicates} duplicated rows')
        return df
        
class DropEmptyColumns(Preprocessor):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)
    
    def run(self, df):
        cols_to_use = [idx for idx,val in (df.isna().mean()>=1.).items() if val==False]
        empty_cols = set(df.columns)-set(cols_to_use)
        df = df.loc[:, cols_to_use]
        self.log(f'Dropped empty columns: {empty_cols}')
        return df

class DropSingleValuedColumns(Preprocessor):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose, **kwargs)

    def run(self, df):
        single_val_cols = [idx for idx,val in df.nunique().items() if val <=1 and df[idx].dtype not in [int, float] ]
        df = df.drop(single_val_cols, axis=1, inplace=False)
        self.log(f'Dropped single-valued columns: {single_val_cols}')
        return df