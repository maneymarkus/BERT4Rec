from absl import logging
import copy
import pathlib
import random
import tensorflow as tf

from bert4rec.dataloaders import dataloader_utils
import bert4rec.evaluation as evaluation
from bert4rec.model import BERTModel, BERT4RecModelWrapper
from bert4rec.model.components import networks
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
        config_path = pathlib.Path(f"../../config/bert_train_configs/{config_identifier}")
        config = utils.load_json_config(config_path)

        bert_encoder = networks.BertEncoder(vocab_size, **config)
        model = BERTModel(bert_encoder)
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

        metrics = self.evaluator.evaluate(
            model_wrapper, prepared_batches, popular_items_ranking=random_popular_items
        )

        self.assertEqual(metrics["valid_ranks"], ds_size,
                         f"The metrics should have as many valid ranks (basically how many ranks have been "
                         f"successfully processed) as there are elements (or better sequences in this case) "
                         f"in the dataset ({ds_size}), but there actually are: {metrics['valid_ranks']}")

        del metrics["valid_ranks"]
        for metric, value in metrics.items():
            self.assertBetween(value, 0, 1,
                               f"Each metric (except valid_ranks) should have a positive float value "
                               f"greater than or equal to 0 and less than or equal to 1, "
                               f"but `{metric}` is `{value}`")

    def test_reset_metrics(self):
        ds_size = 100
        max_seq_len = 5

        prepared_ds, dataloader = test_utils.generate_random_sequence_dataset(ds_size=ds_size, seq_max_len=max_seq_len)
        prepared_batches = dataloader_utils.make_batches(prepared_ds, batch_size=5)
        vocab_size = dataloader.tokenizer.get_vocab_size()

        config_id = "ml-1m_128.json"
        model_wrapper = self._create_model_wrapper(config_id, vocab_size)

        expected_metrics = copy.copy(self.evaluator.get_metrics())

        # usage of set in between to eliminate duplicates
        random_popular_items = list(set([random.randint(0, vocab_size)
                                         for _ in range(200)]))

        metrics = self.evaluator.evaluate(
            model_wrapper, prepared_batches, popular_items_ranking=random_popular_items
        )

        self.assertNotEqual(metrics, expected_metrics,
                            f"After evaluating the metrics should be filled and not in their initial state anymore.")

        self.evaluator.reset_metrics()

        self.assertEqual(self.evaluator.get_metrics(), expected_metrics,
                         f"After resetting the metrics they should be in their initial state again.\n"
                         f"Expected: {expected_metrics}\n"
                         f"Got: {self.evaluator.get_metrics()}")


if __name__ == '__main__':
    tf.test.main()
