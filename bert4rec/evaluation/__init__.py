from bert4rec.evaluation.base_evaluator import BaseEvaluator
from bert4rec.evaluation.bert4rec_evaluator import BERT4RecEvaluator
from bert4rec.evaluation.evaluation_metrics import *


evaluators_map = {
    "bert4rec": BERT4RecEvaluator
}


def get(identifier: str = "bert4rec", **kwargs):
    """
    Factory method to return a concrete evaluator instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if identifier in evaluators_map:
        return evaluators_map[identifier](**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
