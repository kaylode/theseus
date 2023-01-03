import os.path as osp
import pickle

from rank_bm25 import BM25L, BM25Okapi, BM25Plus

from .base import BaseRetrieval


class BM25Retrieval(BaseRetrieval):
    """
    BM25 is a ranking function used by search engines to estimate the relevance of documents
    to a given search query. It is based on the probabilistic retrieval framework
    represent TF-IDF-like retrieval functions used in document retrieval
    """

    def __init__(self, model_name=None, model_path=None):
        if model_path is not None:
            pickle_object = pickle.load(open(model_path, "rb"))
            self.bm25 = pickle_object
        else:
            self.bm25 = None
            self.model_name = model_name

    def encode_corpus(self, corpus):
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        if self.model_name == "BM25Okapi":
            self.bm25 = BM25Okapi(tokenized_corpus)
        elif self.model_name == "BM25L":
            self.bm25 = BM25L(tokenized_corpus)
        elif self.model_name == "BM25Plus":
            self.bm25 = BM25Plus(tokenized_corpus)
        else:
            raise ValueError

    def save_model(self, save_path):
        pickle.dump(self.bm25, open(save_path, "wb"))
        print(f"Save pickle to {save_path}")

    @classmethod
    def from_pretrained(cls, path):
        return cls(model_path=path)

    def get_top_k_similarity(self, querys, corpus=None, top_k=5):
        assert type(querys) == list, "Should be list of texts"

        if self.bm25 is None:
            self.encode_corpus(corpus)

        tokenized_querys = [query.split(" ") for query in querys]

        results = []
        for tokenized_query in tokenized_querys:
            scores = self.bm25.get_scores(tokenized_query).tolist()
            score_mapping = [(i, score) for i, score in enumerate(scores)]
            sorted_scores = sorted(score_mapping, key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            results.append(sorted_scores)
        return results

    def retrieve_similar(self, querys, corpus, top_k=5):
        return self.get_top_k_similarity(querys, corpus, top_k)
