from beir.retrieval.train import TrainRetriever


class TrainAndEvalRetriever(TrainRetriever):

    def init(self, model, batch_size):
        super().__init__(model, batch_size)

    def evaluate(self, evaluator, output_path):
        self.model.evaluate(evaluator=evaluator, output_path=output_path)
