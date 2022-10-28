from bert4rec.trainers.bert4rec_trainer import BERT4RecTrainer


def get(identifier: str = "bert4rec", **kwargs):
    if identifier == "bert4rec":
        return BERT4RecTrainer(**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
