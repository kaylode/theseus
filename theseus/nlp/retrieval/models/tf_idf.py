import os.path as osp
import pickle
from typing import *

from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseRetrieval


def identity_tokenizer(text):
    return text


class TFIDFEncoder(BaseRetrieval):
    def __init__(
        self,
        min_df: int = 0,
        max_df: int = 1.0,
        model_path: str = None,
        ngram_range: Tuple[int] = (1, 1),
    ):

        super().__init__()

        self.min_df = min_df
        self.max_df = max_df

        if model_path is not None:
            pickle_object = pickle.load(open(model_path, "rb"))
            self.tfidfvectorizer = pickle_object
        else:
            self.tfidfvectorizer = TfidfVectorizer(
                analyzer="word",
                min_df=min_df,
                max_df=max_df,
                lowercase=True,
                norm="l2",
                ngram_range=ngram_range,
            )

    def save_model(self, save_path):
        pickle.dump(self.tfidfvectorizer, open(save_path, "wb"))
        dirname = osp.dirname(save_path)
        filename, _ = osp.splitext(osp.basename(save_path))
        with open(osp.join(dirname, f"{filename}_vocab.txt"), "w") as f:
            for term in self.tfidfvectorizer.get_feature_names_out():
                f.write(term + "\n")
        print(f"Save pickle to {save_path}")

    @classmethod
    def from_pretrained(cls, path):
        return cls(model_path=path)

    def encode_corpus(self, corpus):
        return self.tfidfvectorizer.fit_transform(corpus)

    def encode_query(self, query):
        return self.tfidfvectorizer.transform(query)
