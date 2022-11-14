from absl import logging
import tensorflow as tf


from bert4rec.evaluation.evaluation_metrics import *


class EvaluationMetricsTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.ranks_1 = [1, 2, 3, 4, 5]
        self.ranks_2 = [1, 5, 10, 15, 20]
        self.ranks_3 = [2, 8, 4, 13, 20, 6, 3, 11, 2, 5]

    def tearDown(self):
        pass

    def _update_metrics(self, metrics: list[EvaluationMetric], ranks: list[int]):
        for rank in ranks:
            for metric in metrics:
                metric.update(rank)

    def _reset_metrics(self, metrics: list[EvaluationMetric]):
        for metric in metrics:
            metric.reset()

    def test_hit_ratio(self):
        hr1 = HR(1)
        hr5 = HR(5)
        hr10 = HR(10)
        hrs = [hr1, hr5, hr10]

        self._update_metrics(hrs, self.ranks_1)
        self.assertEqual(hr1.result(), 0.2)
        self.assertEqual(hr5.result(), 1)
        self.assertEqual(hr10.result(), 1)
        self._reset_metrics(hrs)

        self._update_metrics(hrs, self.ranks_2)
        self.assertEqual(hr1.result(), 0.2)
        self.assertEqual(hr5.result(), 0.4)
        self.assertEqual(hr10.result(), 0.6)
        self._reset_metrics(hrs)

        self._update_metrics(hrs, self.ranks_3)
        self.assertEqual(hr1.result(), 0)
        self.assertEqual(hr5.result(), 0.5)
        self.assertEqual(hr10.result(), 0.7)
        self._reset_metrics(hrs)

    def test_normalized_discounted_cumulative_gain(self):
        ndcg1 = NDCG(1)
        ndcg5 = NDCG(5)
        ndcg10 = NDCG(10)
        ndcgs = [ndcg1, ndcg5, ndcg10]

        self._update_metrics(ndcgs, self.ranks_1)
        self.assertEqual(ndcg1.result(), 0.2)
        self.assertEqual(round(ndcg5.result(), 2), 0.53)
        self.assertEqual(round(ndcg10.result(), 2), 0.53)
        self._reset_metrics(ndcgs)

        self._update_metrics(ndcgs, self.ranks_2)
        self.assertEqual(ndcg1.result(), 0.2)
        self.assertEqual(round(ndcg5.result(), 2), 0.27)
        self.assertEqual(round(ndcg10.result(), 2), 0.33)
        self._reset_metrics(ndcgs)

        self._update_metrics(ndcgs, self.ranks_3)
        self.assertEqual(ndcg1.result(), 0)
        self.assertEqual(round(ndcg5.result(), 2), 0.22)
        self.assertEqual(round(ndcg10.result(), 2), 0.28)
        self._reset_metrics(ndcgs)

    def test_mean_average_precision(self):
        map = MAP()

        self._update_metrics([map], self.ranks_1)
        self.assertEqual(round(map.result(), 2), 0.46)
        map.reset()

        self._update_metrics([map], self.ranks_2)
        self.assertEqual(round(map.result(), 2), 0.28)
        map.reset()

        self._update_metrics([map], self.ranks_3)
        self.assertEqual(round(map.result(), 2), 0.23)
        map.reset()

    def test_counter(self):
        counter = Counter()

        self._update_metrics([counter], self.ranks_1)
        self.assertEqual(round(counter.result(), 2), 5)
        counter.reset()

        self._update_metrics([counter], self.ranks_2)
        self.assertEqual(round(counter.result(), 2), 5)
        counter.reset()

        self._update_metrics([counter], self.ranks_3)
        self.assertEqual(round(counter.result(), 2), 10)
        counter.reset()

if __name__ == '__main__':
    tf.test.main()
