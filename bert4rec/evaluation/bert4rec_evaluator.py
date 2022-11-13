import tensorflow as tf
import tqdm

from bert4rec.dataloaders import BaseDataloader
from bert4rec.evaluation import evaluation_utils as utils
from bert4rec.evaluation.base_evaluator import BaseEvaluator
from bert4rec.evaluation.evaluation_metrics import *
from bert4rec.model import BERT4RecModelWrapper


bert4rec_evaluation_metrics = [
    NDCG(1),
    NDCG(5),
    NDCG(10),
    HR(1),
    HR(5),
    HR(10),
    MAP()
]


class BERT4RecEvaluator(BaseEvaluator):
    def __init__(self, metrics: list[EvaluationMetric] = None, sample_popular: bool = True):
        if metrics is None:
            metrics = bert4rec_evaluation_metrics
        super().__init__(metrics, sample_popular)

    def evaluate(self, wrapper: BERT4RecModelWrapper,
                 test_data: tf.data.Dataset,
                 dataloader: BaseDataloader = None,
                 popular_items_ranking: list[int] = None) -> list[EvaluationMetric]:

        if popular_items_ranking is None and dataloader is None:
            raise ValueError(f"Either one of the `dataloader` parameter or the `popular_item_ranking` parameter "
                             f"has to be given.")

        if popular_items_ranking is None:
            popular_items_ranking = dataloader.create_popular_item_ranking()

        # iterate over the available batches
        for batch in tqdm.tqdm(test_data):
            self.evaluate_batch(wrapper, batch, popular_items_ranking)

        return self._metrics

    def evaluate_batch(self, wrapper: BERT4RecModelWrapper, test_batch: dict, pop_rank_items: list):
        """
        Evaluation code taken from
        https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run.py#L176

        :param wrapper:
        :param test_batch:
        :param pop_rank_items:
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
                user_rank_items = utils.remove_elements_from_list(pop_rank_items, remove_items)
                ground_truth = selected_mlm_ids[i].numpy()

                # actual sampling "algorithm"
                if self.sample_popular:
                    sampled_rank_items = user_rank_items[:100]
                # random sampling
                else:
                    # first shuffle the sorted rank_items list
                    sampled_rank_items = utils.sample_random_items_from_list(pop_rank_items, 100)

                # append ground truth item id (since this is the one we actually want to rank)
                sampled_rank_items.append(ground_truth)
                ground_truth_items.append(ground_truth)

                rank_item_lists.append(sampled_rank_items)

            rank_item_lists_batch.append(rank_item_lists)
            ground_truth_items_batch.append(ground_truth_items)
            selected_mlm_positions_batch.append(selected_mlm_positions)

        rankings, probabilities = wrapper.rank(test_batch, rank_item_lists_batch, selected_mlm_positions_batch)

        for i, b in enumerate(ground_truth_items_batch):
            for j, idx in enumerate(b):
                rank = np.where(rankings[i][j].numpy() == idx)[0][0]

                # rank is index (starting at 0) so add 1 to get real rank
                rank += 1

                for metric in self._metrics:
                    metric.update(rank)
