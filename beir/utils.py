from beir.retrieval.train import TrainRetriever
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from beir.util import cos_sim
import torch
from tqdm import tqdm, trange
import random
import time
import os
import logging
logger = logging.getLogger(__name__)


class FaissTrainAndEvalRetriever(TrainRetriever):

    def init(self, model, batch_size):
        super().__init__(model, batch_size)

    def evaluate(self, evaluator, output_path):
        self.model.evaluate(evaluator=evaluator, output_path=output_path)

    def load_ir_evaluator(self, corpus, queries, qrels, max_corpus_size=None, name="eval"):

        if len(queries) <= 0:
            raise ValueError("Dev Set Empty!, Cannot evaluate on Dev set.")
        
        rel_docs = {}
        corpus_ids = set()
        
        # need to convert corpus to cid => doc      
        corpus = {idx: corpus[idx].get("title") + " " + corpus[idx].get("text") for idx in corpus}
        
        # need to convert dev_qrels to qid => Set[cid]        
        for query_id, metadata in qrels.items():
            rel_docs[query_id] = set()
            for corpus_id, score in metadata.items():
                if score >= 1:
                    corpus_ids.add(corpus_id)
                    rel_docs[query_id].add(corpus_id)
        
        if max_corpus_size:
            # check if length of corpus_ids > max_corpus_size
            if len(corpus_ids) > max_corpus_size:
                raise ValueError("Your maximum corpus size should atleast contain {} corpus ids".format(len(corpus_ids)))
            
            # Add mandatory corpus documents
            new_corpus = {idx: corpus[idx] for idx in corpus_ids}
            
            # Remove mandatory corpus documents from original corpus
            for corpus_id in corpus_ids:
                corpus.pop(corpus_id, None)
            
            # Sample randomly remaining corpus documents
            for corpus_id in random.sample(list(corpus), max_corpus_size - len(corpus_ids)):
                new_corpus[corpus_id] = corpus[corpus_id]

            corpus = new_corpus

        logger.info("{} set contains {} documents and {} queries".format(name, len(corpus), len(queries)))
        if len(corpus) >= 100000:
            corpus_chunk_size = 100000
        else:
            corpus_chunk_size = len(corpus)
        return FaissInformationRetrievalEvaluator(queries, corpus, rel_docs, corpus_chunk_size=corpus_chunk_size, name=name)


class FaissInformationRetrievalEvaluator(InformationRetrievalEvaluator):

    def init(self, queries, corpus, relevant_docs, corpus_chunk_size, name):
        super().__init__(queries, corpus, relevant_docs, corpus_chunk_size, name)

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):

        # fixed evaluation metrics
        self.score_functions = {"cos_sim": cos_sim}
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.accuracy_at_k = [5, 100]
        self.precision_recall_at_k = [5, 100]

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        # use ndcg instead of map as main metric
        if self.main_score_function is None:
            return max([scores[name]["ndcg@k"][max(self.ndcg_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]["ndcg@k"][max(self.ndcg_at_k)]

    def compute_metrices(self, model, corpus_model=None, corpus_embeddings=None):
        if corpus_model is None:
            corpus_model = model

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        logger.info("Encoding {} queries...".format(len(self.queries)))
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
        # print(f'query embedding shape: {query_embeddings.shape}')

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        logger.info("Encoding {} corpus and computing scores...".format(len(self.corpus)))
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
                # print(f'corpus embedding shape: {sub_corpus_embeddings.shape}')
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():

                # Get top-k values
                # pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
                #
                # #Get top-k values
                # pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                # pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                # pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                # Get top-k values by faiss
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                sub_corpus_embeddings = torch.nn.functional.normalize(sub_corpus_embeddings, p=2, dim=1)
                pair_scores_top_k_values, pair_scores_top_k_idx = self.faiss_topk(query_embeddings.cpu(), 
                                                                                  sub_corpus_embeddings.cpu(), 
                                                                                  max_k)
                pair_scores_top_k_values = pair_scores_top_k_values.tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        # save cosine score
        # import IPython;IPython.embed(colors='linux');exit(1)
        # with open('/tmp2/ttchen/meeting/hard_negative_exp/beir/evalai/nfcorpus.txt', 'w') as f:
        #     for qid, qr in zip(self.queries_ids, queries_result_list['cos_sim']):
        #         for rank, did_score_dict in enumerate(qr):
        #             f.write(f'{qid} Q0 {did_score_dict["corpus_id"]} {rank+1} {did_score_dict["score"]} e5small\n')
        # with open(f'/tmp2/ttchen/meeting/hard_negative_exp/beir/cos_score/scifact/infonce/epoch{epoch}_all_qd_cos_score.tsv', 'w') as f:
        #     cosine_scores = query_embeddings @ sub_corpus_embeddings.T  # [len(queries), len(corpus)]
        #     for i in range(cosine_scores.shape[0]):
        #         for j in range(cosine_scores.shape[1]):
        #             if self.corpus_ids[j] in self.relevant_docs[self.queries_ids[i]]:
        #                 f.write(f'{self.queries_ids[i]}\t{self.corpus_ids[j]}\t{cosine_scores[i][j].item()}\t1\n')
        #             else:
        #                 f.write(f'{self.queries_ids[i]}\t{self.corpus_ids[j]}\t{cosine_scores[i][j].item()}\t0\n')

        return scores

    def faiss_topk(self, query_embeddings, sub_corpus_embeddings, k):
        import faiss
        index = faiss.IndexFlatIP(len(query_embeddings[0]))  # dimension of embedding
        index.add(sub_corpus_embeddings)
        distance, idx = index.search(query_embeddings, min(k, len(sub_corpus_embeddings)))
        return distance, idx


