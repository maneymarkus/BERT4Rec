from bert4rec.evaluation.bert4rec_evaluator import BERT4RecEvaluator


class EvaluatorFactory:
    def get_evaluator(self, model: str = "bert4rec", **kwargs):
        if model == "bert4rec":
            return BERT4RecEvaluator(**kwargs)
        else:
            raise ValueError(f"{model} is not known!")


evaluator_factory = EvaluatorFactory()
