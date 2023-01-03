from typing import Any, Dict, List, Optional

import numpy as np
import sentence_transformers.util as sentfms_utils
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base import BaseRetrieval


class RetrievalModel(BaseRetrieval):
    def __init__(
        self,
        config_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ):

        super().__init__()
        self.model = SentenceTransformer(config_name)
        self.device = device
        if device:
            self.model = self.model.to(self.device)

    def gen_embeddings(self, sentences):
        # query should be converted to true format
        with torch.no_grad():
            outputs = self.model.encode(sentences)
        return outputs

    def encode(self, corpus, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(corpus), batch_size)):
            batch_corpus = self.gen_embeddings(corpus[i : i + batch_size])
            embeddings.append(batch_corpus)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def encode_query(self, corpus):
        return self.encode(corpus, batch_size=512)

    def encode_corpus(self, corpus):
        return self.encode(corpus, batch_size=512)
