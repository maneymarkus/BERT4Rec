import abc
import json
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import BaseDataloader
from bert4rec.evaluation.evaluation_metrics import EvaluationMetric
from bert4rec.models import BERT4RecModelWrapper


class BaseEvaluator(abc.ABC):
    def __init__(self, metrics: list[EvaluationMetric], sample_popular: bool = True):
        self.sample_popular = sample_popular
        self._metrics = metrics
        self.reset_metrics()

    def reset_metrics(self) -> None:
        for metric in self._metrics:
            metric.reset()

    @abc.abstractmethod
    def evaluate(self, wrapper: BERT4RecModelWrapper, test_data: tf.data.Dataset, dataloader: BaseDataloader) \
            -> list[EvaluationMetric]:
        """
        Evaluates a given model on the given test_data. The dataloader provides the method
        `create_popular_item_ranking()` on which the negative sampling is based on.

        :param wrapper:
        :param test_data:
        :param dataloader:
        :return: dict containing the results of the metrics
        """
        pass

    def get_metrics(self) -> list[EvaluationMetric]:
        return self._metrics

    def get_metrics_results(self) -> dict:
        metrics_dict = dict()

        for metric in self._metrics:
            metrics_dict[metric.name] = metric.result()

        return metrics_dict

    def save_results(self, save_path: pathlib.Path) -> pathlib.Path:
        """
        Save the results (or the current status of the metrics) to the given save_path

        :param save_path:
        :return:
        """
        if save_path.is_dir():
            save_path = save_path.joinpath("eval_results.json")

        metrics_dict = self.get_metrics_results()

        with open(save_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        return save_path
