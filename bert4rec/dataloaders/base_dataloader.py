import abc
import tensorflow as tf

from bert4rec.tokenizers.base_tokenizer import BaseTokenizer


class BaseDataloader(abc.ABC):
    def __int__(self, tokenizer: BaseTokenizer = None):
        self.tokenizer = tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    @property
    @abc.abstractmethod
    def dataset_code(self):
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
    def prepare_training(self):
        """
        Prepares the represented dataset completely for training. This includes generating the vocab
        for the tokenizer, splitting the dataset into train, validation and test parts, preparing parts
        of the training data for finetuning, making batches,

        :return:
        """
        pass

    @abc.abstractmethod
    def generate_vocab(self, source=None) -> True:
        """
        Fills the vocab of the tokenizer with items from the respective dataset or given source

        :return: True
        """
        pass

    @abc.abstractmethod
    def create_popular_item_ranking(self) -> list:
        pass
