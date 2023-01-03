import os
import os.path as osp
import pickle

import numpy as np
import scipy
import sentence_transformers.util as sentfms_utils
import torch
from sklearn.metrics.pairwise import linear_kernel


class BaseRetrieval:
    def __init__(self, *args, **kwargs):
        pass

    def encode_query(self, query):
        raise NotImplementedError

    def encode_corpus(self, corpus):
        raise NotImplementedError

    def save_embeddings(self, embeddings, outpath):
        folder_name = osp.dirname(outpath)
        os.makedirs(folder_name, exist_ok=True)
        pickle.dump(embeddings, open(outpath, "wb"))
        print(f"Save pickle to {outpath}")

    @staticmethod
    def load_embeddings(path):
        return pickle.load(open(path, "rb"))

    def get_top_k_similarity(self, encoded_query, encoded_corpus, top_k=5):
        results = []
        if isinstance(encoded_query, scipy.sparse.csr_matrix):
            cosine_scores = linear_kernel(encoded_query, encoded_corpus)
            for cosine_score in cosine_scores:
                cosine_score = list(enumerate(cosine_score))
                cosine_score = sorted(cosine_score, key=lambda x: x[1], reverse=True)
                top_k_relevants = cosine_score[:top_k]
                results.append(top_k_relevants)
        else:
            outputs = sentfms_utils.semantic_search(
                query_embeddings=encoded_query,
                corpus_embeddings=encoded_corpus,
                top_k=top_k,
            )
            for batch in outputs:
                tmp = []
                for item in batch:
                    tmp.append((item["corpus_id"], item["score"]))
                results.append(tmp)

        return results

    def retrieve_similar(self, querys, corpus, top_k=5):
        if isinstance(querys, list):
            encoded_query = self.encode_query(querys)
        elif isinstance(querys, str):  ## load from pickle path
            encoded_query = self.load_embeddings(
                querys
            )  # return encoded querys and query ids
        else:
            raise ValueError()

        if isinstance(corpus, list):
            encoded_corpus = self.encode_corpus(corpus)
        elif isinstance(corpus, str):  ## load from pickle path
            encoded_corpus = self.load_embeddings(
                corpus
            )  # return encoded querys and query ids
        else:
            raise ValueError()

        if isinstance(encoded_query, np.ndarray):
            encoded_query = torch.from_numpy(encoded_query)
            if self.device:
                encoded_query = encoded_query.to(self.device)
        if isinstance(encoded_corpus, np.ndarray):
            encoded_corpus = torch.from_numpy(encoded_corpus)
            if self.device:
                encoded_corpus = encoded_corpus.to(self.device)

        return self.get_top_k_similarity(encoded_query, encoded_corpus, top_k=top_k)
