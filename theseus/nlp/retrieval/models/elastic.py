import time

from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Search

from .base import BaseRetrieval


class ElasticSearch(BaseRetrieval):
    def __init__(self, index_name):
        super().__init__()
        self.index_name = index_name
        self.es = Elasticsearch(
            ["http://0.0.0.0:9200"],
            timeout=400,
            connection_class=RequestsHttpConnection,
            http_auth=("elastic", "123456"),
            use_ssl=False,
            verify_certs=False,
        )

        time.sleep(10)

    def flush(self):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

    def encode_corpus(self, corpus):
        def gen_data():
            for idx, doc in enumerate(corpus):
                yield {
                    "_index": self.index_name,
                    "_id": f"{idx}",
                    "_source": {"id": idx, "texts": doc.split(" ")},
                }

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, ignore=400)
            self.es.indices.get_alias("*")
            bulk(self.es, gen_data())
        else:
            print("Corpus already indexed")

    def retrieve_similar(self, querys, corpus, top_k=5):
        if not self.es.indices.exists(index=self.index_name):
            self.encode_corpus(corpus)

        results = []
        for query in querys:
            responses = []
            query = query.lower()
            s = (
                Search(using=self.es, index=self.index_name)
                .query("multi_match", query=query, fields=["texts"])
                .extra(size=top_k, explain=True)
            )
            responses = s.execute()

            results.append((responses.hit.id, responses.hit.meta.score))
            print(results)
            asdf

        return results
