import abc
from absl import logging
import tensorflow as tf
from typing import Union

from bert4rec import tokenizers
from bert4rec.dataloaders import dataloader_utils as utils


class BaseDataloader(abc.ABC):
    def __init__(self, tokenizer: Union[str, tokenizers.BaseTokenizer] = None):
        self.tokenizer = tokenizers.get(tokenizer)

    def get_tokenizer(self):
        return self.tokenizer

    @property
    @abc.abstractmethod
    def dataset_identifier(self):
        pass

    @abc.abstractmethod
    def load_data_into_ds(self) -> tf.data.Dataset:
        """
        Load data into a tf.data.Dataset object

        :return: A (unprocessed) dataset object from the loaded sources
        """
        pass

    @abc.abstractmethod
    def load_data_into_split_ds(self) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """
        Loads data and already splits them in three separate tf.data.Dataset objects (train, validation and test)

        :return: Up to three (unprocessed) dataset objects from the loaded sources
        """
        pass

    @abc.abstractmethod
    def preprocess_dataset(self, ds: tf.data.Dataset = None) -> tf.data.Dataset:
        """
        Preprocesses the dataset to prepare it for usage with the model. Param `ds` is optional as each
        dataloader is only responsible for a single dataset and therefore only ever loads and preprocesses one dataset

        :param ds: The dataset that should be preprocessed
        :return: The preprocessed dataset
        """
        pass

    @abc.abstractmethod
    def prepare_training(self, finetuning_split: float = 0.1):
        """
        Prepares the represented dataset completely for training. This includes generating the vocab
        for the tokenizer, splitting the dataset into train, validation and test parts, preparing parts
        of the training data for finetuning, making batches,

        :param finetuning_split: Determines if and what share of the train dataset should be
            preprocessed for finetuning purposes
        :return:
        """
        pass

    @abc.abstractmethod
    def prepare_inference(self, sequence):
        pass

    @abc.abstractmethod
    def generate_vocab(self, source=None) -> True:
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
