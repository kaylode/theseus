
from .base import Preprocessor


class PreprocessCompose(Preprocessor):
    def __init__(self, preproc_list: list[Preprocessor], **kwargs):
        self.preproc_list = preproc_list

    def run(self, df):
        for preproc in self.preproc_list:
            df = preproc.run(df)

        return df
