from typing import List
from .base import Preprocessor

class PreprocessCompose(Preprocessor):
    def __init__(self, preproc_list: List[Preprocessor], **kwargs):
        self.preproc_list = preproc_list 

    def run(self, df):
        for preproc in self.preproc_list:
            df = preproc.run(df)

        return df
        