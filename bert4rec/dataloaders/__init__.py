import abc

from .dataloader_utils import *
from .base_dataloader import BaseDataloader
from .bert4rec_dataloader import BERT4RecDataloader
from .bert4rec_ml1m_dataloader import BERT4RecML1MDataloader
from .bert4rec_ml20m_dataloader import BERT4RecML20MDataloader
from .bert4rec_beauty_dataloader import BERT4RecBeautyDataloader
from .bert4rec_steam_dataloader import BERT4RecSteamDataloader
from .bert4rec_reddit_dataloader import BERT4RecRedditDataloader


class BaseDataloaderFactory(abc.ABC):
    @abc.abstractmethod
    def create_ml_1m_dataloader(self, **kwargs) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_ml_20m_dataloader(self, **kwargs) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_beauty_dataloader(self, **kwargs) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_steam_dataloader(self, **kwargs) -> BaseDataloader:
        pass

    @abc.abstractmethod
    def create_reddit_dataloader(self, **kwargs) -> BaseDataloader:
        pass


############################################################
# Concrete Dataloader Factories                            #
############################################################

class BERT4RecDataloaderFactory(BaseDataloaderFactory):
    def create_ml_1m_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecML1MDataloader(**kwargs)

    def create_ml_20m_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecML20MDataloader(**kwargs)

    def create_beauty_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecBeautyDataloader(**kwargs)

    def create_steam_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecSteamDataloader(**kwargs)

    def create_reddit_dataloader(self, **kwargs) -> BERT4RecDataloader:
        return BERT4RecRedditDataloader(**kwargs)


def get_dataloader_factory(identifier: str = "bert4rec") -> BaseDataloaderFactory:
    if identifier == "bert4rec":
        return BERT4RecDataloaderFactory()
    else:
        raise ValueError(f"{identifier} is not a known model/identifier!")
