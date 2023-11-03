import spacy

from .base import BaseRetrieval


class SpacyEncoder(BaseRetrieval):
    def __init__(self, name) -> None:
        super().__init__()
        # "en_core_sci_lg" "en_core_med7_lg"
        self.encoder = spacy.load(name)

    def encode_corpus(self, text):
        return self.encoder(text).vector

    def encode_query(self, text):
        return self.encoder(text).vector
