from sentence_transformers import losses, SentenceTransformer
from utils import InBatchTripletLoss, MixupMultipleNegativesRankingLoss, InfoNCELoss, NegOnlyMultipleNegativesRankingLoss, BCELoss
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.train import TrainRetriever
from utils import FaissTrainAndEvalRetriever
import pathlib
import os
import logging
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-trd", default="nq-train_train", type=str)
    parser.add_argument("--test_dataset", "-ted", default="nq_test", type=str)
    parser.add_argument("--hard_negative_sample", "-hns", default=None, type=str)
    parser.add_argument("--model_name", "-m", default="intfloat/e5-small", type=str)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--batch_size", "-bs", default=16, type=int)
    parser.add_argument("--loss", "-l", default='infonce', type=str)
    parser.add_argument("--scale", "-scale", default=20, type=float)
    parser.add_argument("--margin", "-margin", default=0, type=float)
    args = parser.parse_args()

    # Print information
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    # Load training data
    if not args.hard_negative_sample:  # random hard negative
        if args.train_dataset != args.test_dataset:
            dataset, split = args.train_dataset.split("_")
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
            out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
            data_path = util.download_and_unzip(url, out_dir)
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        else:  # finetune on test data for debug
            dataset, split = args.train_dataset.split("_")
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
            out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
            data_path = util.download_and_unzip(url, out_dir)
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    else:  # load hard negative triplets: [(query, pos_text, hard_neg_text)]
        with open(args.hard_negative_sample, 'r') as file:
            triplets = [json.loads(line) for line in file]
        

    # Load testing data
    dataset, split = args.test_dataset.split("_")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split=split)

    # Pretrained sentence-transformer model
    model = SentenceTransformer(args.model_name)
    # retriever = TrainRetriever(model=model, batch_size=args.batch_size)
    retriever = FaissTrainAndEvalRetriever(model=model, batch_size=args.batch_size)

    # Prepare training samples
    if not args.hard_negative_sample:  # random negative
        train_samples = retriever.load_train(corpus, queries, qrels)
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    else:  # load hard negative triplets: [(query, pos_text, hard_neg_text)]
        train_samples = retriever.load_train_triplets(triplets)
        train_dataloader = retriever.prepare_train_triplets(train_samples)

    # Training with cosine-similarity
    if args.loss == 'triplet':  # broken
        train_loss = InBatchTripletLoss(model=retriever.model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=1)
    elif args.loss == 'mixup':
        train_loss = MixupMultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.cos_sim)
    elif args.loss =='infonce' and args.margin==0:
        train_loss = InfoNCELoss(model=retriever.model, similarity_fct=util.cos_sim)
    elif args.loss =='infonce' and args.margin:
        train_loss = InfoNCELoss(model=retriever.model, similarity_fct=util.cos_sim, margin=args.margin)
    elif args.loss =='negonly':  # broken
        train_dataloader = retriever.prepare_train(train_samples, shuffle=False)
        train_loss = NegOnlyMultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.cos_sim)
    elif args.loss == 'bce' and args.margin==0:
        train_loss = BCELoss(model=retriever.model)
    elif args.loss == 'bce' and args.margin:
        train_loss = BCELoss(model=retriever.model, margin=args.margin)

    # Prepare dev evaluator
    ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

    # Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-{}-{}{}-{}epochs".format("e5-small", args.test_dataset, args.loss, args.margin, args.epochs))
    os.makedirs(model_save_path, exist_ok=True)

    # Configure Train params
    num_epochs = args.epochs
    evaluation_steps = 9999999  # never evaluate during an epoch
    evaluation_steps = 5000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    if num_epochs == 0:
        # Evaluate the pretrained model
        retriever.evaluate(evaluator=ir_evaluator, output_path=model_save_path)
    else:
        # Finetuned and evaluate the pretrained model
        retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=ir_evaluator,
                      epochs=num_epochs,
                      output_path=model_save_path,
                      warmup_steps=warmup_steps,
                      evaluation_steps=evaluation_steps,
                      use_amp=True)
