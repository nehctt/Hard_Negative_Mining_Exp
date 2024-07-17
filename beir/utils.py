from beir.retrieval.train import TrainRetriever
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from beir.util import cos_sim
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import ModelCardTemplate, fullname, batch_to_device
import torch
from tqdm import tqdm, trange
import random
import time
import os
import json
import logging
logger = logging.getLogger(__name__)


class FaissTrainAndEvalRetriever(TrainRetriever):

    def __init__(self, model, batch_size=16, save_corpus_emb=False):
        super().__init__(model=model, batch_size=batch_size)
        # if True, we save corpus embedding when evaluation
        self.save_corpus_emb = save_corpus_emb

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
        return FaissInformationRetrievalEvaluator(queries, corpus, rel_docs, corpus_chunk_size=corpus_chunk_size, name=name, save_corpus_emb=self.save_corpus_emb)


class FaissInformationRetrievalEvaluator(InformationRetrievalEvaluator):

    def __init__(self, queries, corpus, relevant_docs, corpus_chunk_size, name = '', save_corpus_emb=False):
        super().__init__(queries=queries, corpus=corpus, relevant_docs=relevant_docs, corpus_chunk_size=corpus_chunk_size, name=name)    
        # if True, we save corpus embedding when evaluation
        self.save_corpus_emb = save_corpus_emb

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):

        # fixed evaluation metrics
        self.score_functions = {"cos_sim": cos_sim}
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.accuracy_at_k = [5, 100]
        self.precision_recall_at_k = [5, 100]

        self.csv_headers = ["epoch at"]
        for score_name in self.score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in self.precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in self.mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in self.ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            for k in self.map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model, save_corpus_emb=self.save_corpus_emb, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            # if not os.path.isfile(csv_path):
            if epoch == 0:
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch]
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

    def compute_metrices(self, model, save_corpus_emb=False, corpus_model=None, corpus_embeddings=None):
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

                # pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
                #
                # Get top-k values
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

            # Save corpus embedding
            # import IPython;IPython.embed(colors='linux');exit(1)
            if save_corpus_emb:
                corpus_emb_path = './hard_negative_sampler/dynamic/ance/corpus_embeddings/corpus_embs.tsv'
                if corpus_start_idx==0:
                    f_corpus_emb = open(corpus_emb_path, 'w')
                else:
                    f_corpus_emb = open(corpus_emb_path, 'a')
                for sub_corpus_id, emb in zip(self.corpus_ids[corpus_start_idx:corpus_end_idx], sub_corpus_embeddings):
                    f_corpus_emb.write(f'{sub_corpus_id}\t{emb.tolist()}\n')


        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        return scores

    def faiss_topk(self, query_embeddings, sub_corpus_embeddings, k):
        import faiss
        index = faiss.IndexFlatIP(len(query_embeddings[0]))  # dimension of embedding
        index.add(sub_corpus_embeddings)
        distance, idx = index.search(query_embeddings, min(k, len(sub_corpus_embeddings)))
        return distance, idx


class MySentenceTransformer(SentenceTransformer):
    ''' to adjust training steps of multi-task learning '''

    def fit(self,
            train_objectives,
            evaluator = None,
            epochs = 1,
            steps_per_epoch = None,
            scheduler = 'WarmupLinear',
            warmup_steps = 10000,
            optimizer_class = torch.optim.AdamW,
            optimizer_params = {'lr': 2e-5},
            weight_decay = 0.01,
            evaluation_steps = 0,
            output_path = None,
            save_best_model = True,
            max_grad_norm = 1,
            use_amp = False,
            callback = None,
            show_progress_bar = True,
            checkpoint_path = None,
            checkpoint_save_steps = 500,
            checkpoint_save_total_limit = 0
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """

        ##Add info to model card
        #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            # modify for multi-task learning
            # steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
            steps_per_epoch = max([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        steps_each_dataloader = [len(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            # modify for multi-task learning
            for train_idx in range(num_train_objectives):
                data_iterator = iter(dataloaders[train_idx])
                data_iterators[train_idx] = data_iterator

            for step in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    if (steps_per_epoch - steps_each_dataloader[train_idx]) > step:
                        continue
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        continue

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
