from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import random
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", "-trd", default="nfcorpus", type=str)
parser.add_argument("--model_name", "-m", default="intfloat/e5-small", type=str)
args = parser.parse_args()
print(f'use {args.model_name} to get margin>(pos+neg)/2 hard negative sample on {args.train_dataset}')


#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.train_dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "../../datasets/"
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

#### Prepare dataset texts
queries_ids = list(queries.keys())
queries = [queries[qid] for qid in queries_ids]
corpus_ids = list(corpus.keys())
corpus = [corpus[cid]["title"] + " " + corpus[cid]["text"] for cid in corpus_ids]

#### Pretrained sentence-transformer model
model = SentenceTransformer(args.model_name)

#### Get query & corpus embeddings
query_embeddings = model.encode(queries)
corpus_embeddings = model.encode(corpus)

#### Get cosine scores and negative samples
cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)
print(f'mean of cos_scores (shape: {cos_scores.shape}): {cos_scores.mean()}')
triplets = []
num_hard_negatives = 5

for i in tqdm(range(cos_scores.shape[0])):
    pos_docs = [doc_id for doc_id in qrels[queries_ids[i]]]
    pos_doc_texts = [corpus[corpus_ids.index(doc_id)] for doc_id in pos_docs]
    # use (pos_mean - neg_mean) as margin
    pos_scores = np.array([cos_scores[i][corpus_ids.index(doc_id)] for doc_id in pos_docs])
    pos_mean_score = pos_scores.mean()
    neg_scores = cos_scores[i].sum() - pos_scores.sum()
    neg_mean_score = (neg_scores / (cos_scores.shape[1] - len(pos_scores))).item()
    mean_score = (pos_mean_score + neg_mean_score) / 2
    for pos_text in pos_doc_texts:
        random_sample = random.sample(range(cos_scores.shape[1]), cos_scores.shape[1])
        # random_sample = list(range(cos_scores.shape[1]))
        # random.shuffle(random_sample)
        negative_sample_count = 0
        for j in random_sample:
            if corpus_ids[j] in qrels[queries_ids[i]].keys():
                continue
            if mean_score > cos_scores[i][j]:
                continue
            if negative_sample_count == num_hard_negatives:
                break
            neg_text = corpus[j]
            triplets.append((queries[i], pos_text, neg_text))
            negative_sample_count += 1

print(f'There are {len(triplets)} triplets in total')
with open(f'../../datasets/{dataset}-hns/margin2_{num_hard_negatives}.jsonl', 'w') as file:
    for item in triplets:
        json.dump(item, file)
        file.write('\n')
