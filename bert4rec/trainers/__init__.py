from bert4rec.trainers.base_trainer import BaseTrainer
from bert4rec.trainers.bert4rec_trainer import BERT4RecTrainer


trainers_map = {
    "bert4rec": BERT4RecTrainer
}


def get(identifier: str = "bert4rec", **kwargs) -> BaseTrainer:
    """
    Factory method to return a concrete trainer instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if identifier in trainers_map:
        return trainers_map[identifier](**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
