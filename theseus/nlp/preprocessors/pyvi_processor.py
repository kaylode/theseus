from .base import BaseProcessor
from pyvi import ViTokenizer
# https://github.com/explosion/spaCy/tree/master/spacy/lang/vi

class PyviProcessor(BaseProcessor):
    def __call__(self, text):
        text = ViTokenizer.tokenize(text)
        return text