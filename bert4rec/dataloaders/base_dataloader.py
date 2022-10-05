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
    def load_data(self) -> tf.data.Dataset:
        """
        Load data into a tf.data.Dataset object

        :return:
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
    def generate_vocab(self) -> True:
        """
        Fills the vocab of the tokenizer with items from the respective dataset

        :return: True
        """
        pass

    @abc.abstractmethod
    def create_popular_item_ranking(self) -> list:
        pass
