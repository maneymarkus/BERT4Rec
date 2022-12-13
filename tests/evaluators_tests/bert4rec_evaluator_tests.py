from absl import logging
import copy
import random
import tensorflow as tf

from bert4rec.dataloaders import dataloader_utils
import bert4rec.evaluation as evaluation
from bert4rec.evaluation import evaluation_metrics
from bert4rec.models import BERT4RecModel, BERT4RecModelWrapper
from bert4rec.models.components import networks
import bert4rec.utils as utils
import tests.test_utils as test_utils


class Bert4RecEvaluatorTest(tf.test.TestCase):
    def setUp(self):
        super(Bert4RecEvaluatorTest, self).setUp()
        logging.set_verbosity(logging.DEBUG)
        self.evaluator = evaluation.get()

    def tearDown(self):
        self.evaluator = None

    def _create_model_wrapper(self, config_identifier: str = "ml-1m_128.json", vocab_size: int = 1000):
        # load a specific config
        config_path = utils.get_project_root().joinpath(f"config/bert_train_configs/{config_identifier}")
        config = utils.load_json_config(config_path)

        bert_encoder = networks.Bert4RecEncoder(vocab_size, **config)
        model = BERT4RecModel(bert_encoder)
        model_wrapper = BERT4RecModelWrapper(model)
        # makes sure the weights are built
        _ = model(model.inputs)

        return model_wrapper

    def test_evaluate_batch(self):
        ds_size = 1000
        max_seq_len = 10

        prepared_ds, dataloader = test_utils.generate_random_sequence_dataset(ds_size=ds_size, seq_max_len=max_seq_len)
        prepared_batches = dataloader_utils.make_batches(prepared_ds, batch_size=5)
        vocab_size = dataloader.tokenizer.get_vocab_size()

        config_id = "ml-1m_128.json"
        model_wrapper = self._create_model_wrapper(config_id, vocab_size)

        # usage of set in between to eliminate duplicates
        random_popular_items = list(set([random.randint(0, vocab_size)
                                         for _ in range(200)]))

        metric_objects = self.evaluator.evaluate(
            model_wrapper, prepared_batches, tokenized_ds_item_list=random_popular_items
        )

        metrics = self.evaluator.get_metrics_results()

        self.assertEqual(metrics["Valid Ranks"], ds_size,
                         f"The metrics should have as many valid ranks (basically how many ranks have been "
                         f"successfully processed) as there are elements (or better sequences in this case) "
                         f"in the dataset ({ds_size}), but there actually are: {metrics['Valid Ranks']}")

        del metrics["Valid Ranks"]
        for metric, value in metrics.items():
            self.assertBetween(value, 0, 1,
                               f"Each metric (except valid_ranks) should have a positive float value "
                               f"greater than or equal to 0 and less than or equal to 1, "
                               f"but `{metric}` is `{value}`")

    def test_reset_metrics(self):
        ds_size = 100
        max_seq_len = 5

        metrics = [
            evaluation_metrics.Counter(),
            evaluation_metrics.HR(100),
            evaluation_metrics.NDCG(100)
        ]

        self.evaluator = evaluation.get(**{"metrics": metrics})

        prepared_ds, dataloader = test_utils.generate_random_sequence_dataset(ds_size=ds_size, seq_max_len=max_seq_len)
        prepared_batches = dataloader_utils.make_batches(prepared_ds, batch_size=5)
        vocab_size = dataloader.tokenizer.get_vocab_size()

        config_id = "ml-1m_128.json"
        model_wrapper = self._create_model_wrapper(config_id, vocab_size)

        initial_metrics = [
            copy.copy(metric_object) for metric_object in self.evaluator.get_metrics()
        ]

        # usage of set in between to eliminate duplicates
        random_popular_items = list(set([random.randint(0, vocab_size)
                                         for _ in range(200)]))

        metrics = self.evaluator.evaluate(
            model_wrapper, prepared_batches, tokenized_ds_item_list=random_popular_items
        )

        for i, metric in enumerate(metrics):
            self.assertNotEqual(metric.result(), initial_metrics[i].result(),
                                f"After evaluating the metrics should be filled and not in their initial state "
                                f"anymore.")

        self.evaluator.reset_metrics()

        for i, metric in enumerate(self.evaluator.get_metrics()):
            self.assertEqual(metric.result(), initial_metrics[i].result(),
                             f"After resetting the metrics they should be in their initial state again.\n"
                             f"Expected: {initial_metrics}\n"
                             f"Got: {self.evaluator.get_metrics()}")


if __name__ == '__main__':
    tf.test.main()
