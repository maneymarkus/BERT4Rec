from bert4rec.evaluation.base_evaluator import BaseEvaluator
from bert4rec.evaluation.bert4rec_evaluator import BERT4RecEvaluator


def get(identifier: str = "bert4rec", **kwargs):
    """
    Factory method to return a concrete evaluator instance

    :param identifier:
    :param kwargs:
    :return:
    """
    if identifier == "bert4rec":
        return BERT4RecEvaluator(**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
