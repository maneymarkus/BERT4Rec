from abc import ABC, abstractmethod

from .dataloader_utils import *


class BaseDataloaderFactory(ABC):
    @abstractmethod
    def create_ml_1m_dataloader(self):
        pass

    @abstractmethod
    def create_ml_20m_dataloader(self):
        pass

    @abstractmethod
    def create_reddit_dataloader(self):
        pass

    @abstractmethod
    def create_imdb_dataloader(self):
        pass


class BaseDataloader(ABC):
    @abstractmethod
    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        pass


############################################################
# Concrete Dataloader Factories                            #
############################################################

class BERT4RecDataloaderFactory(BaseDataloaderFactory):
    def create_ml_1m_dataloader(self):
        pass

    def create_ml_20m_dataloader(self):
        pass

    def create_reddit_dataloader(self):
        pass

    def create_imdb_dataloader(self):
        pass


def get_dataloader_factory(model: str = "bert4rec") -> BaseDataloaderFactory:
    if model == "bert4rec":
        return BERT4RecDataloaderFactory()
    else:
        raise ValueError(f"{model} is not a known model!")
