from absl import logging
import tensorflow as tf
import tqdm
from typing import Union

from bert4rec.dataloaders import BaseDataloader, samplers
from bert4rec.evaluation.base_evaluator import BaseEvaluator
from bert4rec.evaluation.evaluation_metrics import *
from bert4rec.models import BERT4RecModel


bert4rec_evaluation_metrics = [
    Counter(name="Valid Ranks"),
    NDCG(1),
    NDCG(5),
    NDCG(10),
    HR(1),
    HR(5),
    HR(10),
    MAP()
]


class BERT4RecEvaluator(BaseEvaluator):
    def __init__(self,
                 metrics: list[EvaluationMetric] = None,
                 sampler: Union[str, samplers.BaseSampler] = "pop_random",
                 dataloader: BaseDataloader = None):
        if metrics is None:
            metrics = bert4rec_evaluation_metrics
        if isinstance(sampler, str):
            sampler_config = {
                "sample_size": 100
            }
            if dataloader is not None:
                vocab = dataloader.tokenizer.get_vocab()
                tokenized_vocab = dataloader.tokenizer.tokenize(vocab)
                sampler_config.update({
                    "source": dataloader.create_item_list_tokenized(),
                    "vocab": tokenized_vocab
                })
            sampler = samplers.get(sampler, **sampler_config)

        super().__init__(metrics, sampler, dataloader)

    def evaluate(self,
                 model: BERT4RecModel,
                 test_data: tf.data.Dataset) -> list[EvaluationMetric]:

        if self.dataloader is None and not self.sampler.is_fully_prepared():
            raise ValueError("The evaluator has to be either initialized with a dataloader or "
                             "a fully prepared sampler has to be given.")

        # iterate over the available batches
        for batch in tqdm.tqdm(test_data, total=test_data.cardinality().numpy()):
            self.evaluate_batch(model, batch)

        return self._metrics

    def evaluate_batch(self, model: BERT4RecModel, test_batch: dict):
        """
        Evaluation code inspired from
        https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run.py#L176

        :param model:
        :param test_batch:
        :return:
        """

        rank_item_lists_batch = []
        ground_truth_items_batch = []
        selected_mlm_positions_batch = []

        # iterate over a single batch (containing the batched data)
        for t_i, masked_lm_positions in enumerate(test_batch["masked_lm_positions"]):
            rank_item_lists = []
            ground_truth_items = []

            # only select "weighted" items -> generate boolean mask from int tensor by using tf.cast()
            masked_lm_weights = tf.cast(test_batch["masked_lm_weights"][t_i], tf.bool)
            # use ragged boolean mask to preserve shape(s)
            selected_mlm_positions = tf.ragged.boolean_mask(masked_lm_positions,
                                                            masked_lm_weights)
            selected_mlm_ids = tf.ragged.boolean_mask(test_batch["masked_lm_ids"][t_i],
                                                      masked_lm_weights)

            # iterate over the tokens that should serve as the "rank source"
            for i in range(len(selected_mlm_positions)):
                # negative sampling
                # remove all items from the list of items to be ranked that the user has already interacted with
                remove_items = test_batch["labels"][t_i].numpy().tolist()
                ground_truth = selected_mlm_ids[i].numpy()
                # remove ground truth item from sample as well to consistently add it afterwards and always
                # use 100 + 1 (the ground truth item) items to rank
                remove_items.append(ground_truth)

                # sample items to rank
                sampled_rank_items = self.sampler.sample(without=remove_items)

                # append ground truth item id (since this is the one we actually want to rank)
                sampled_rank_items.append(ground_truth)
                ground_truth_items.append(ground_truth)

                rank_item_lists.append(sampled_rank_items)

            rank_item_lists_batch.append(rank_item_lists)
            ground_truth_items_batch.append(ground_truth_items)
            selected_mlm_positions_batch.append(selected_mlm_positions)

        rankings = model.rank_items(test_batch, rank_item_lists_batch)

        for i, b in enumerate(ground_truth_items_batch):
            for j, idx in enumerate(b):
                rank = np.where(rankings[i][j].numpy() == idx)[0][0]

                # rank is index (starting at 0) so add 1 to get real rank
                rank += 1

                for metric in self._metrics:
                    metric.update(rank)
