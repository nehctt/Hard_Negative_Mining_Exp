from beir.datasets.data_loader import GenericDataLoader
from beir import util
from sentence_transformers import SentenceTransformer
import torch
import csv
import ast
import faiss
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def faiss_topk(query_embeddings, sub_corpus_embeddings, k):
    index = faiss.IndexFlatIP(len(query_embeddings[0]))  # dimension of embedding
    index.add(sub_corpus_embeddings)
    distance, idx = index.search(query_embeddings, min(k, len(sub_corpus_embeddings)))
    return distance, idx


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-trd", default="nfcorpus", type=str)
    parser.add_argument("--model_name", "-m", default="intfloat/e5-small", type=str)
    parser.add_argument("--warmup", "-warmup", default=False, type=bool)
    parser.add_argument("--num_negative_samples", "-n", default=5, type=int)
    args = parser.parse_args()
    print(f'get ADORE hard negative sample on {args.train_dataset}')

    # Download nfcorpus.zip dataset and unzip the dataset
    dataset = args.train_dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "../../datasets/"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

    # Pretrained sentence-transformer model
    model = SentenceTransformer(args.model_name)

    # if first epoch, we have to save corpus emb first
    if 'e5-small' in args.model_name:
        model_path = 'intfloat-e5-small'
    if 'mean-tokens' in args.model_name:
        model_path = 'bert-base-nli-stsb-mean-tokens'
    if 'uncased' in args.model_name:
        model_path = 'bert-base-uncased'

    corpus_emb_path = f"../corpus_embeddings_e0/{dataset}/{model_path}/corpus_embs.tsv"
    if args.warmup:
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["title"] + " " + corpus[cid]["text"] for cid in corpus_ids]
        chunksize = 100000
        for corpus_start_idx in tqdm(range(0, len(corpus), chunksize), desc='Encoding Corpus'):
            corpus_end_idx = min(corpus_start_idx + chunksize, len(corpus))
            sub_corpus_embeddings = model.encode(corpus_texts[corpus_start_idx:corpus_end_idx], convert_to_tensor=True)

            Path(corpus_emb_path).parent.mkdir(parents=True, exist_ok=True)
            if corpus_start_idx==0:
                f_corpus_emb = open(corpus_emb_path, 'w')
            else:
                f_corpus_emb = open(corpus_emb_path, 'a')
            for sub_corpus_id, emb in zip(corpus_ids[corpus_start_idx:corpus_end_idx], sub_corpus_embeddings):
                f_corpus_emb.write(f'{sub_corpus_id}\t{emb.tolist()}\n')

    # Prepare dataset texts
    queries_ids = list(queries.keys())
    pos_doc_ids = []
    pos_doc_texts = []
    query_ids = []  # new query_ids for relavant docs
    for q_id in tqdm(queries_ids):
        pos_doc_ids_temp = [doc_id for doc_id in qrels[q_id] if qrels[q_id][doc_id] > 0]
        pos_doc_ids += pos_doc_ids_temp
        pos_doc_texts_temp = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_doc_ids_temp]
        pos_doc_texts += pos_doc_texts_temp
        query_ids_temp = [q_id for _ in pos_doc_ids_temp]
        query_ids += query_ids_temp

    # Get pos_doc embeddings
    pos_doc_embeddings = model.encode(pos_doc_texts, convert_to_tensor=True, show_progress_bar=True)
    pos_doc_embeddings = torch.nn.functional.normalize(pos_doc_embeddings, p=2, dim=1)
    print(f'The shape of training sample emb: {pos_doc_embeddings.shape}')
    print(f'The number of corpus: {len(corpus)}')

    # Read corpus embeddings and get top-k hard negative
    file_path = corpus_emb_path

    chunksize = 100000
    result_list = [[] for _ in range(len(pos_doc_embeddings))]
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        sub_corpus_ids = []
        sub_corpus_embs = []
        for row in tqdm(reader, total=len(corpus), desc='Reading Corpus Embeddings'):
            sub_corpus_ids.append(row[0])
            sub_corpus_embs.append(ast.literal_eval(row[1]))

            if len(sub_corpus_ids) >= chunksize:
                # get top-k ANCE hard negative
                topk_scores, topk_ids = faiss_topk(pos_doc_embeddings.cpu(), torch.tensor(sub_corpus_embs).cpu(), 100)
                # print(topk_scores, topk_ids)

                for pos_doc_itr in range(len(pos_doc_embeddings)):
                    for sub_corpus_id_idx, score in zip(topk_ids[pos_doc_itr], topk_scores[pos_doc_itr]):
                        corpus_id = sub_corpus_ids[sub_corpus_id_idx]
                        result_list[pos_doc_itr].append({'corpus_id': corpus_id, 'score': score})

                sub_corpus_ids = []
                sub_corpus_embs = []

        if len(sub_corpus_ids) > 0:
            # print('last chunk')
            # get top-k ANCE hard negative
            topk_scores, topk_ids = faiss_topk(pos_doc_embeddings.cpu(), torch.tensor(sub_corpus_embs).cpu(), 100)
            # print(topk_scores, topk_ids)

            for pos_doc_itr in range(len(pos_doc_embeddings)):
                for sub_corpus_id_idx, score in zip(topk_ids[pos_doc_itr], topk_scores[pos_doc_itr]):
                    corpus_id = sub_corpus_ids[sub_corpus_id_idx]
                    result_list[pos_doc_itr].append({'corpus_id': corpus_id, 'score': score})

    triplets = []
    for query_itr in tqdm(range(len(result_list)), desc='Selecting Hard Negatives'):
        query_id = query_ids[query_itr]
        query_text = queries[query_id]
        pos_text = pos_doc_texts[query_itr]

        # Sort scores
        top_hits = sorted(result_list[query_itr], key=lambda x: x['score'], reverse=True)[:100]
        neg_ids = []
        for result in top_hits:
            neg_id = result['corpus_id']
            if neg_id in qrels[query_id]:
                continue
            else:
                neg_ids.append(neg_id)

            if len(neg_ids) == args.num_negative_samples:
                break

        for neg_id in neg_ids:
            neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
            triplets.append((query_text, pos_text, neg_text))

    # print(triplets[:5])
    print(f'There are total {len(triplets)} ADORE hard negative training triplets')

    ### Save training triplets
    save_path = f'../../../datasets/{dataset}-hns/adore_{args.num_negative_samples}.jsonl'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as file:
        for item in triplets:
            json.dump(item, file)
            file.write('\n')
