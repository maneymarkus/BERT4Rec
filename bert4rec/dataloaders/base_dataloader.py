import abc
from absl import logging
import tensorflow as tf
from typing import Union, Type

from bert4rec import tokenizers
from bert4rec.dataloaders import dataloader_utils as utils
from bert4rec.dataloaders.preprocessors import BasePreprocessor
from datasets import BaseDataset


class BaseDataloader(abc.ABC):
    def __init__(self,
                 tokenizer: Union[str, tokenizers.BaseTokenizer] = None,
                 data_source: Type[BaseDataset] = None,
                 preprocessor: Type[BasePreprocessor] = None,
                 **kwargs):
        self.tokenizer = tokenizers.get(tokenizer)
        self.data_source = data_source
        self.preprocessor = preprocessor

    def get_tokenizer(self):
        return self.tokenizer

    @property
    @abc.abstractmethod
    def dataset_identifier(self):
        pass

    @abc.abstractmethod
    def get_data(self,
                 split_data: bool = False,
                 sort_by: str = None,
                 extract_data: list = [],
                 datatypes: list = [],
                 duplication_factor: int = None,
                 **kwargs) \
            -> tuple:
        """
        Loads data from a "raw" source into a Dataset and performs preprocessing operations on it
        This is the main method to load and already preprocess data with the most control
        via method parameters

        :param split_data: Determines if the data should be split (into training, validation and
            testing data). Depending on the given value, the tuple returned will either
            contain a single dataset or three
            dataset(s)
        :param sort_by: If given, sorts the dataframe by this column prior to extracting anything
        :param extract_data: The names of the columns that should be extracted into to returned
            dataset(s)
        :param datatypes: A list of datatypes describing the pd.DataFrame column values.
            Relevant for converting the DataFrame columns into datasets. Datatypes may be tried to
            be inferred but especially for sequence data (lists) this argument is necessary
        :param duplication_factor: If given, determines how many times the (training) dataset
            should be duplicated
        """
        if duplication_factor is not None and duplication_factor < 1:
            raise ValueError(f"A duplication factor of less than 1 (given: {duplication_factor}) "
                             "is not allowed!")
        datasets = self.load_data(split_data, sort_by, extract_data, datatypes, duplication_factor)
        processed_datasets = tuple()
        for ds in datasets:
            processed_datasets += (self.preprocessor.process_dataset(ds),)
        return processed_datasets

    @abc.abstractmethod
    def load_data(self,
                  split_data: bool = False,
                  sort_by: str = None,
                  extract_data: list = [],
                  datatypes: list = [],
                  duplication_factor: int = None,
                  **kwargs) \
            -> tuple:
        """
        Load data into a tf.data.Dataset object (no preprocessing yet)

        :return: A (unprocessed) dataset object from the loaded sources
        """
        pass

    @abc.abstractmethod
    def process_data(self, ds: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        """
        Apply dataloader individual preprocessing operations on the given dataset
        """
        pass

    @abc.abstractmethod
    def prepare_training(self, **kwargs) -> tuple:
        """
        Prepares the represented dataset completely for training. This includes generating the
        vocab for the tokenizer, splitting the dataset into train, validation and test parts
        and preparing parts of the training data for finetuning (if wanted)

        :return:
        """
        pass

    @abc.abstractmethod
    def prepare_inference(self, data):
        """
        Prepare given data for inference. The given data should be completely unprocessed, so
        not even tokenized or anything. In terms of recommendation tasks the data would at least
        consist of a sequence of items (i.e. list of strings)
        """
        pass

    @abc.abstractmethod
    def generate_vocab(self, source=None, progress_bar: bool = True) -> True:
        """
        Fills the vocab of the tokenizer with items from the respective dataset or given source

        :return: True
        """
        pass

    @abc.abstractmethod
    def create_item_list(self) -> list:
        pass

    def create_item_list_tokenized(self) -> list:
        logging.info("Create dataset item list")
        item_list = self.create_item_list()
        logging.info(f"Start tokenizing dataloader item list (len: {len(item_list)})")
        return self.tokenizer.tokenize(item_list, progress_bar=True)

    def create_popular_item_ranking(self) -> list:
        item_list = self.create_item_list()
        return utils.rank_items_by_popularity(item_list)

    def create_popular_item_ranking_tokenized(self) -> list:
        sorted_item_list = self.create_popular_item_ranking()
        return self.tokenizer.tokenize(sorted_item_list)
