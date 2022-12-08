import fnmatch 

class FilterColumnNames():
    '''
    Return all columns name match the filter
    '''
    def __init__(self, patterns) -> None:
        self.patterns = patterns

    def run(self, df):
        filtered_columns = []
        for pattern in self.patterns:
            filtered_columns += fnmatch.filter(df.columns, pattern)

        filtered_columns = list(set(filtered_columns))
        return filtered_columns