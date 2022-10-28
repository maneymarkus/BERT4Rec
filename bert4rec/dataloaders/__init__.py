import abc

from .dataloader_utils import *
from .base_dataloader import BaseDataloader
from .bert4rec_dataloaders import BERT4RecDataloader, BERT4RecML1MDataloader, BERT4RecML20MDataloader, \
    BERT4RecIMDBDataloader, BERT4RecRedditDataloader


class BaseDataloaderFactory(abc.ABC):
    @abc.abstractmethod
    def create_ml_1m_dataloader(self) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_ml_20m_dataloader(self) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_reddit_dataloader(self) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_imdb_dataloader(self) -> BaseDataloader:
        pass


############################################################
# Concrete Dataloader Factories                            #
############################################################

class BERT4RecDataloaderFactory(BaseDataloaderFactory):
    def create_ml_1m_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecML1MDataloader(**kwargs)

    def create_ml_20m_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecML20MDataloader(**kwargs)

    def create_reddit_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecRedditDataloader(**kwargs)

    def create_imdb_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecIMDBDataloader(**kwargs)


def get_dataloader_factory(identifier: str = "bert4rec") -> BaseDataloaderFactory:
    if identifier == "bert4rec":
        return BERT4RecDataloaderFactory()
    else:
        raise ValueError(f"{identifier} is not a known model/identifier!")
