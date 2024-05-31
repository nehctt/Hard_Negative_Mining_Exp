from sentence_transformers import losses, SentenceTransformer
from loss import MixupMultipleNegativesRankingLoss, InfoNCELoss, BCELoss, DCLLoss, InfoNCEDynamicMarginLoss
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.train import TrainRetriever
from utils import FaissTrainAndEvalRetriever, MySentenceTransformer
import pathlib
import os
import json
import pandas as pd
import logging
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-trd", default="nq-train_train", type=str)  # multi task ex.: "nfcorpus+scifact_multi"
    parser.add_argument("--test_dataset", "-ted", default="nq_test", type=str)
    parser.add_argument("--hard_negative_sample", "-hns", default=None, type=str)  # {"random", "bm25", ...}
    parser.add_argument("--num_hns", "-nhns", default=5, type=int)
    parser.add_argument("--model_name", "-m", default="intfloat/e5-small", type=str)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--batch_size", "-bs", default=16, type=int)
    parser.add_argument("--loss", "-l", default='infonce', type=str)
    parser.add_argument("--scale", "-scale", default=20, type=float)
    args = parser.parse_args()

    # Print information
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)


    # Pretrained sentence-transformer model
    # model = SentenceTransformer(args.model_name)
    model = MySentenceTransformer(args.model_name)
    # retriever = TrainRetriever(model=model, batch_size=args.batch_size)
    if args.hard_negative_sample == 'ance':
        retriever = FaissTrainAndEvalRetriever(model=model, batch_size=args.batch_size, save_corpus_emb=True)
    else:
        retriever = FaissTrainAndEvalRetriever(model=model, batch_size=args.batch_size)


    # Load training data
    if not args.hard_negative_sample:  # random negative or DEBUG by fine-tuning on test dataset
        train_dataloaders = []
        dataset, split = args.train_dataset.split("_")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        # Prepare training samples
        train_samples = retriever.load_train(corpus, queries, qrels)
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
        train_dataloaders.append(train_dataloader)
    else:  # load hard negative triplets: [(query, pos_text, hard_neg_text)]
        dataset, split = args.train_dataset.split("_")
        train_dataloaders = []
        if split != "multi":  # if single task(dataset)
            hns_path = os.path.join(
                pathlib.Path(__file__).parent.absolute(),
                "datasets",
                f"{dataset}-hns",
                f"{args.hard_negative_sample}_{str(args.num_hns)}.jsonl"
            )
            with open(hns_path, 'r') as file:
                triplets = [json.loads(line) for line in file]
            # Prepare training triplets
            train_samples = retriever.load_train_triplets(triplets)
            train_dataloader = retriever.prepare_train_triplets(train_samples)
            train_dataloaders.append(train_dataloader)
        if split == "multi":  # multi task(dataset) learning
            multi_data = dataset.split("+")
            for data in multi_data:
                hns_path = os.path.join(
                    pathlib.Path(__file__).parent.absolute(),
                    "datasets",
                    f"{data}-hns",
                    f"{args.hard_negative_sample}_{str(args.num_hns)}.jsonl"
                )
                with open(hns_path, 'r') as file:
                    triplets = [json.loads(line) for line in file]
                # Prepare training triplets
                train_samples = retriever.load_train_triplets(triplets)
                train_dataloader = retriever.prepare_train_triplets(train_samples)
                train_dataloaders.append(train_dataloader)
        

    # Load testing data
    dataset, split = args.test_dataset.split("_")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split=split)


    # Prepare test evaluator
    ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)


    # Training with cosine-similarity
    if args.loss =='infonce':
        train_loss = InfoNCELoss(model=retriever.model, similarity_fct=util.cos_sim)
    if args.loss =='infoncedm':
        train_loss = InfoNCEDynamicMarginLoss(model=retriever.model, similarity_fct=util.cos_sim)
    if args.loss == 'bce':
        train_loss = BCELoss(model=retriever.model)
    if args.loss == 'mixup':
        train_loss = MixupMultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.cos_sim)
    if args.loss == 'dcl':
        train_loss = DCLLoss(model=retriever.model, similarity_fct=util.cos_sim)


    # Provide model save path
    if args.model_name[:6] != "output":  # loading hf pretrained model
        if not args.hard_negative_sample:
            model_save_path = os.path.join(
                pathlib.Path(__file__).parent.absolute(),
                "output",
                f"{dataset}_{args.model_name.replace('/', '-')}_{args.loss}_{args.epochs}epochs"
            )
        else:
            model_save_path = os.path.join(
                pathlib.Path(__file__).parent.absolute(),
                "output",
                f"{dataset}_{args.hard_negative_sample}{args.num_hns}_{args.model_name.replace('/', '-')}_{args.loss}_{args.epochs}epochs"
            )
    else:  # loading our local fine-tuned model
        # the model name is the model save path
        model_save_path = args.model_name
    os.makedirs(model_save_path, exist_ok=True)


    # Configure Train params
    num_epochs = args.epochs
    evaluation_steps = 9999999  # never evaluate during an epoch
    # evaluation_steps = 10000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    if num_epochs == 0:
        # Evaluate the pretrained model
        retriever.evaluate(evaluator=ir_evaluator, output_path=model_save_path)
    else:
        if len(train_dataloaders) > 1:
            logger.info(f"Multi-Task Training with {len(train_dataloaders)} Tasks")
        # Finetuned and evaluate the pretrained model
        retriever.fit(train_objectives=[(train_dataloader, train_loss) for train_dataloader in train_dataloaders],
                      evaluator=ir_evaluator,
                      epochs=num_epochs,
                      # steps_per_epoch=  how to define multitask step_per_epoch
                      output_path=model_save_path,
                      warmup_steps=warmup_steps,
                      evaluation_steps=evaluation_steps,
                      use_amp=True)

        # read eval csv and return result
        result = pd.read_csv(f'{model_save_path}/eval/Information-Retrieval_evaluation_eval_results.csv')
        best_row = result[result['cos_sim-NDCG@10'] == result['cos_sim-NDCG@10'].max()]
        print("\nFinish training. The best result:")
        for row in best_row.itertuples(index=False, name=None):
            for col_name, value in zip(best_row.columns, row):
                print(f"{col_name: <24}{round(value, 4)}")
        print("\n")


    # if dynamic is true
