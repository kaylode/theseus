from pyvi import ViTokenizer

from .base import BaseProcessor

# https://github.com/explosion/spaCy/tree/master/spacy/lang/vi


class PyviProcessor(BaseProcessor):
    def __call__(self, text):
        text = ViTokenizer.tokenize(text)
        return text
