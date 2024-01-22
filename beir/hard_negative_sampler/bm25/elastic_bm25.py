from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
import tqdm
import json


#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "../../datasets/"
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

#### https://www.elastic.co/
hostname = "localhost"
index_name = "nfcorpus"

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
hard_negatives_max = 10

#### Retrieve BM25 hard negatives => Given a positive document, find most similar lexical documents
no_hits_count = 0
for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25"):
    query_id, query_text = qids[idx], queries[qids[idx]]
    pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
    pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
    try:
        hits = bm25.retriever.es.lexical_multisearch(texts=pos_doc_texts, top_hits=hard_negatives_max+1)
        for (pos_text, hit) in zip(pos_doc_texts, hits):
            for (neg_id, _) in hit.get("hits"):
                if neg_id not in pos_docs:
                    neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
                    triplets.append((query_text, pos_text, neg_text))
    except:
        no_hits_count += 1
print(f'There are {no_hits_count} queries has no BM25 results')
print(f'There are total {len(triplets)} hard negative training triplets')

### Save training triplets
with open('../../datasets/nfcorpus-hns/bm25.jsonl', 'w') as file:
    for item in triplets:
        json.dump(item, file)
        file.write('\n')
