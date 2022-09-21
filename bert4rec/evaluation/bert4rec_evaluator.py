import copy
import numpy as np
import random
import tensorflow as tf

from bert4rec.dataloaders import BaseDataloader
from bert4rec.evaluation import evaluation_utils as utils
from bert4rec.evaluation.base_evaluation import BaseEvaluation
from bert4rec.model import BERT4RecModelWrapper


class BERT4RecEvaluator(BaseEvaluation):
    def __init__(self, sample_popular: bool = True):
        super().__init__(sample_popular)
        self.valid_ranks = 0
        self.ndcg_1_count = 0
        self.ndcg_1 = 0.0
        self.hit_1_count = 0
        self.hit_1 = 0.0
        self.ndcg_5_count = 0
        self.ndcg_5 = 0.0
        self.hit_5_count = 0
        self.hit_5 = 0.0
        self.ndcg_10_count = 0
        self.ndcg_10 = 0.0
        self.hit_10_count = 0
        self.hit_10 = 0.0
        self.ap_count = 0.0
        self.ap = 0.0

        self.metrics.update({
            "valid_ranks": self.valid_ranks,
            "ndcg@1": self.ndcg_1,
            "hit@1": self.hit_1,
            "ndcg@5": self.ndcg_5,
            "hit@5": self.hit_5,
            "ndcg@10": self.ndcg_10,
            "hit@10": self.hit_10,
            "ap": self.ap
        })

    def evaluate(self, wrapper: BERT4RecModelWrapper, test_data: tf.data.Dataset, dataloader: BaseDataloader) -> dict:
        pop_rank_items = dataloader.create_popular_item_ranking()
        # iterate over the available batches
        for batch in test_data:
            self.evaluate_batch(wrapper, batch, pop_rank_items)

        self.ndcg_1 = self.ndcg_1_count / self.valid_ranks
        self.hit_1 = self.hit_1_count / self.valid_ranks
        self.ndcg_5 = self.ndcg_5_count / self.valid_ranks
        self.hit_5 = self.hit_5_count / self.valid_ranks
        self.ndcg_10 = self.ndcg_10_count / self.valid_ranks
        self.hit_10 = self.hit_10_count / self.valid_ranks
        self.ap = self.ap_count / self.valid_ranks
        self.metrics.update({
            "valid_ranks": self.valid_ranks,
            "ndcg@1": self.ndcg_1,
            "hit@1": self.hit_1,
            "ndcg@5": self.ndcg_5,
            "hit@5": self.hit_5,
            "ndcg@10": self.ndcg_10,
            "hit@10": self.hit_10,
            "ap": self.ap
        })

        return self.metrics

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

        # iterate over a single batch (containing the batched data)
        for t_i, tensor in enumerate(test_batch["masked_lm_positions"]):
            rank_item_lists = []
            ground_truth_items = []

            # iterate over the tokens that should serve as the "rank source"
            for i in range(len(test_batch["masked_lm_positions"][t_i])):
                # negative sampling
                # remove all items from the list of items to be ranked that the user has already interacted with
                remove_items = test_batch["labels"][t_i].numpy().tolist()
                user_rank_items = utils.remove_elements_from_list(pop_rank_items, remove_items)
                ground_truth = test_batch["masked_lm_ids"][t_i][i].numpy()

                # actual sampling "algorithm"
                if self.sample_popular:
                    sampled_rank_items = user_rank_items[:100]
                # random sampling
                else:
                    # first shuffle the sorted rank_items list
                    random_items = copy.copy(pop_rank_items)
                    random.shuffle(random_items)
                    sampled_rank_items = [random.choice(pop_rank_items) for _ in range(100)]

                # append ground truth item id (since this is the one we actually want to rank)
                sampled_rank_items.append(ground_truth)
                ground_truth_items.append(ground_truth)

                rank_item_lists.append(sampled_rank_items)

            rank_item_lists_batch.append(rank_item_lists)
            ground_truth_items_batch.append(ground_truth_items)

        rankings, probabilities = wrapper.rank(test_batch, rank_item_lists_batch, test_batch["masked_lm_positions"])

        for i, b in enumerate(ground_truth_items_batch):
            for j, idx in enumerate(b):
                rank = np.where(rankings[i][j].numpy() == idx)[0][0]

                self.valid_ranks += 1

                if rank < 1:
                    self.ndcg_1_count += 1
                    self.hit_1_count += 1
                if rank < 5:
                    self.ndcg_5_count += 1 / np.log2(rank + 2)
                    self.hit_5_count += 1
                if rank < 10:
                    self.ndcg_10_count += 1 / np.log2(rank + 2)
                    self.hit_10_count += 1
                self.ap_count += 1 / (rank + 1)