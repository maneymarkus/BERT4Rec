import abc
from absl import logging
import json
import pathlib
import tensorflow as tf
from typing import Union

from bert4rec.dataloaders.base_dataloader import BaseDataloader
from bert4rec.dataloaders import samplers
from bert4rec.evaluation.evaluation_metrics import EvaluationMetric
from bert4rec.models.model_wrapper import ModelWrapper


class BaseEvaluator(abc.ABC):
    def __init__(self,
                 metrics: list[EvaluationMetric],
                 sampler: Union[str, samplers.BaseSampler] = "popular",
                 dataloader: BaseDataloader = None):
        self.sampler = samplers.get(sampler)

        if self.sampler.sample_size is None:
            logging.warning(f"The sampler used in the evaluator {self} does not have a sample size "
                            "set. This might lead to problems during evaluation.")

        if self.sampler.source is None:
            logging.info(f"The sampler used in the evaluator {self} does not have a source set. "
                         "To execute evaluation make sure to either set a source when initializing "
                         "the sampler or provide a source list when calling the evaluate() method "
                         "of the evaluator.")

        if dataloader is not None:
            self.sampler.set_source(dataloader.create_item_list_tokenized())

        self._metrics = metrics
        self.dataloader = dataloader
        self.reset_metrics()

    def reset_metrics(self) -> None:
        for metric in self._metrics:
            metric.reset()

    @abc.abstractmethod
    def evaluate(self, wrapper: ModelWrapper,
                 test_data: tf.data.Dataset,
                 tokenized_ds_item_list: list[int] = None) \
            -> list[EvaluationMetric]:
        """
        Evaluates a given model on the given test_data. The dataloader provides the method
        `create_popular_item_ranking()` on which the negative sampling is based on.

        :param wrapper:
        :param test_data:
        :param tokenized_ds_item_list:
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
