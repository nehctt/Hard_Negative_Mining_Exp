from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
import random
import math
import json
import tqdm


#### SimANS https://github.com/microsoft/SimXNS/blob/main/SimANS/README.md
def SimANS(pos_pair, neg_pair_list, num_hard_negatives):
    pos_id, pos_score = pos_pair[0], pos_pair[1]
    neg_candidates, neg_scores = [], []
    for pair in neg_pair_list:
        neg_id, neg_score = pair
        neg_score = math.exp(-(neg_score - pos_score) ** 2 * 0.5)
        neg_candidates.append(neg_id)
        neg_scores.append(neg_score)
    return random.choices(neg_candidates, weights=neg_scores, k=num_hard_negatives)


#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nq-train"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "../../datasets/"
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

#### https://www.elastic.co/
hostname = "localhost"
index_name = "nq"

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
bm25 = EvaluateRetrieval(model)

#### Index passages into the index (seperately)
bm25.retriever.index(corpus)

triplets = []
qids = list(qrels) 
k = 100  # build distribution in top-k results
hard_negatives_max = 5

#### Retrieve BM25 hard negatives => Given a positive document, find most similar lexical documents
no_hits_count = 0
no_pos_count = 0
for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25 with SimANS"):
    query_id, query_text = qids[idx], queries[qids[idx]]
    pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
    pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
    try:
        hits = bm25.retriever.es.lexical_search(text=query_text, top_hits=k+1)
        # Get pos_pair=(pos_text, pos_score) and neg_pair_list=[(neg_text, neg_score), ...]
        for (pos_id, pos_text) in zip(pos_docs, pos_doc_texts):
            pos_pair = [pos_id]
            neg_pair_list = []
            for rank, (neg_id, neg_score) in enumerate(hits.get("hits")):
                if neg_id != pos_pair[0]:
                    neg_pair_list.append([neg_id, neg_score])
                else:
                    pos_pair.append(neg_score)
                    # print(f'{pos_id} with score: {neg_score} and rank: {rank+1}')
            if len(pos_pair) > 1:  # if postive sample in top 200 bm25 results
                neg_ids = SimANS(pos_pair, neg_pair_list, hard_negatives_max)
                # print(neg_ids)
                for neg_id in neg_ids:
                    neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
                    triplets.append((query_text, pos_text, neg_text))
            else:
                no_pos_count += 1

    except:
        no_hits_count += 1
print(f'There are {no_hits_count} queries has no BM25 results')
print(f'There are {no_pos_count} postive_text has no show in top-{k} BM25 results')
print(f'There are total {len(triplets)} hard negative training triplets')

### Save training triplets
with open('../../datasets/nq-hns/bm25_simans_5.jsonl', 'w') as file:
    for item in triplets:
        json.dump(item, file)
        file.write('\n')
