from .base import BaseRetrieval
import numpy as np

class EnsembleRetriever(BaseRetrieval):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ensemble_cosine_matrix(self, cosines, top_k=5):
        '''
        List of sorted cosine matrices
        '''

        num_cosines = len(cosines)
        num_querys, corpus_size = cosines[0].shape
        results = {
            query_id: {
                corpus_id: [] for corpus_id in range(corpus_size)
            }
            for query_id in range(num_querys)
        }

        for cosine_id in range(num_cosines):
            for query_id in range(num_querys):
                for corpus_id in range(corpus_size):
                    results[query_id][corpus_id].append(
                        cosines[cosine_id][query_id][corpus_id]
                    )

        final_results = []
        for query_id in range(num_querys):
            for corpus_id in range(corpus_size):
                results[query_id][corpus_id] = np.mean(results[query_id][corpus_id])

            sorted_score_query = sorted(results[query_id].items(), key = lambda x: x[1], reverse=True)
            final_results.append(sorted_score_query[:top_k])

        return final_results

    def ensemble_prediction(self, predictions, top_k=5):
        # prediction: (methods, num_query, num_answers_per_query)
        num_methods = len(predictions)
        results = {
            i: {} for i in range(len(predictions[0]))
        }
        for preds_per_method in predictions:
            for query_id, pred in enumerate(preds_per_method):
                for top_results in pred:
                    pred_id, retrieved_score = top_results
                    if pred_id not in results[query_id].keys():
                        results[query_id][pred_id] = []
                    results[query_id][pred_id].append(retrieved_score)

        final_results = []
        for query_id in results.keys():
            for pred_id in results[query_id].keys():
                results[query_id][pred_id] = sum(results[query_id][pred_id]) / num_methods

            sorted_score_query = sorted(results[query_id].items(), key = lambda x: x[1], reverse=True)
            final_results.append(sorted_score_query[:top_k])

        return final_results

    def retrieve_similar(self, querys, corpus, additional_scores=None, top_k=5):
        '''
        querys and corpus should be list of pickles, for ensemble 
        '''
        cosine_score_ensemble = []
        for query_pickle, corpus_pickle in zip(querys, corpus):
            
            encoded_query = self.load_embeddings(query_pickle) 
            encoded_corpus = self.load_embeddings(corpus_pickle)
            cosine_scores = self.get_top_k_similarity(encoded_query, encoded_corpus, top_k=300)
            cosine_score_ensemble.append(cosine_scores)

        ## For those method that cannot generate embeddings, ranking predictions can be used
        if additional_scores is not None:
            assert type(additional_scores) == list, 'Should be list of predictions'
            for cosine_scores in additional_scores:
                cosine_score_ensemble.append(cosine_scores)

        results = self.ensemble_prediction(cosine_score_ensemble, top_k=top_k)
        return results


if __name__ == '__main__':
    from dataset import ZaloAI22Dataset, ZaloAI22TestDataset
    data_pool = ZaloAI22Dataset(
        '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/data/e2eqa-train+public_test-v1/zac2022_train_merged_final.json',
        append_question_mark=True).corpus
    questions, question_ids = ZaloAI22TestDataset(
        '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/data/e2eqa-train+public_test-v1/zac2022_testa_only_question.json', 
        append_question_mark=True).corpus

    ensembler = EnsembleRetriever()
    results = ensembler.retrieve_similar(
        corpus = [
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_sev_pretrained_train.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_multilingual_pretrained_train.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_vsbert_pretrained_train.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_phobert_pretrained_train.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfidf_train.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfidf_pyvi_train.pickle',
        ],
        querys = [
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_sev_pretrained_test_a.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_multilingual_pretrained_test_a.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_vsbert_pretrained_test_a.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfms_phobert_pretrained_test_a.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfidf_test_a.pickle',
            '/mnt/4TBSSD/pmkhoi/zaloai/source/e2e_qa/outputs/embeddings/tfidf_pyvi_test_a.pickle',
        ],
        top_k=3
    )

    for question, result in zip(questions[:5], results[:5]):
        print('>>>>>', question)
        for ret in result:
            print(round(ret[1],3), ': ', data_pool[ret[0]])