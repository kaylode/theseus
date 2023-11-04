import fnmatch


class FilterColumnNames:
    """
    Return all columns name match the filter
    """

    def __init__(self, patterns, excludes=None) -> None:
        self.patterns = patterns
        self.excludes = excludes

    def run(self, df):
        filtered_columns = []
        for pattern in self.patterns:
            filtered_columns += fnmatch.filter(df.columns, pattern)
        filtered_columns = set(filtered_columns)
        if self.excludes:
            for exclude in self.excludes:
                filtered_columns -= set(fnmatch.filter(df.columns, exclude))
        return list(filtered_columns)
