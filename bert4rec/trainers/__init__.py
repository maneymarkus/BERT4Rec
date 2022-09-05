from bert4rec.trainers.bert4rec_trainer import BERT4RecTrainer


class TrainerFactory:
    def get_trainer(self, model: str = "bert4rec", **kwargs):
        if model == "bert4rec":
            return BERT4RecTrainer(**kwargs)
        else:
            raise ValueError(f"{model} is not known!")


trainer_factory = TrainerFactory()
