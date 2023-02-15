import abc
from typing import Type

from absl import logging
import pandas as pd
import pathlib


class BaseDataset(abc.ABC):
    """

    Attributes:
        load_n_records (int): Allows to specify an integer variable that may limit the amount of
            data records that should be loaded from the source.
        source (str): Defines the source from where to download the dataset
        dest (pathlib.Path): Defines the desired local path to put the dataset

    """
    load_n_records: int = None
    source: str = None
    dest: pathlib.Path = None

    def __int__(self, n_records: int = None):
        self.load_n_records = n_records

    @classmethod
    def load_data(cls) -> pd.DataFrame:
        """
        Loads all available data of the represented dataset into a pd.DataFrame object
        """
        if not cls.is_available():
            logging.info("Dataset is not available (yet).")
            logging.info("Download and unpack dataset.")
            cls.download()

        logging.info("Load data into pd.DataFrame.")

        if not cls.dest.exists():
            raise RuntimeError(f"Something went wrong. There are no data at {cls.dest}")

        return cls.extract_data()

    @classmethod
    def set_load_n_records(cls, n_records: int) -> Type["BaseDataset"]:
        cls.load_n_records = n_records
        return cls

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def download(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def extract_data(cls) -> pd.DataFrame:
        pass
