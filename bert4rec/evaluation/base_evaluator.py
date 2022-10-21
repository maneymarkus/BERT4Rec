import abc
import json
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import BaseDataloader
from bert4rec.model import BERT4RecModelWrapper


class BaseEvaluator(abc.ABC):
    def __init__(self, sample_popular: bool = True):
        self.sample_popular = sample_popular
        self.metrics = dict()

    @abc.abstractmethod
    def evaluate(self, wrapper: BERT4RecModelWrapper, test_data: tf.data.Dataset, dataloader: BaseDataloader) -> dict:
        """
        Evaluates a given model on the given test_data. The dataloader provides the method
        `create_popular_item_ranking()` on which the negative sampling is based on.

        :param wrapper:
        :param test_data:
        :param dataloader:
        :return: dict containing the results of the metrics
        """
        pass

    def save_results(self, save_path: pathlib.Path):
        with open(save_path.joinpath("eval_results.txt"), "w") as f:
            json.dump(self.metrics, f, indent=4)
