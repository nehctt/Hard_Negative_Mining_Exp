from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
import random
import json
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-trd", default="nfcorpus", type=str)
    parser.add_argument("--num_negative_samples", "-n", default=5, type=int)
    args = parser.parse_args()


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

    triplets = []
    num_hard_negatives = args.num_negative_samples

    for i in tqdm(range(len(queries))):
        pos_docs = [doc_id for doc_id in qrels[queries_ids[i]]]
        pos_doc_texts = [corpus[corpus_ids.index(doc_id)] for doc_id in pos_docs]
        for pos_text in pos_doc_texts:
            random_sample = random.sample(range(len(corpus)), 50)
            negative_sample_count = 0
            for j in random_sample:
                if corpus_ids[j] in qrels[queries_ids[i]].keys():
                    continue
                if negative_sample_count == num_hard_negatives:
                    break
                neg_text = corpus[j]
                triplets.append((queries[i], pos_text, neg_text))
                negative_sample_count += 1

    print(f'There are {len(triplets)} random triplets in total')
    save_path = f'../../datasets/{dataset}-hns/random_{num_hard_negatives}.jsonl'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as file:
        for item in triplets:
            json.dump(item, file)
            file.write('\n')
