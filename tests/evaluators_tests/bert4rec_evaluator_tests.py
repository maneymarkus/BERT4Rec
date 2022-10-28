import pathlib
import random
import tensorflow as tf

from bert4rec.dataloaders import BERT4RecDataloader, dataloader_utils
import bert4rec.evaluation as evaluation
from bert4rec.model import BERTModel, BERT4RecModelWrapper
from bert4rec.model.components import networks
import bert4rec.utils as utils
import tests.test_utils as test_utils


class Bert4RecEvaluatorTest(tf.test.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _create_test_dataset(self, ds_size: int = 1000, seq_min_len: int = 5, seq_max_len: int = 100):
        subject_list = []
        sequence_list = []
        for i in range(ds_size):
            subject_list.append(random.randint(0, ds_size * 2))
            sequence_length = random.randint(seq_min_len, seq_max_len)
            sequence = test_utils.generate_random_word_list(size=sequence_length)
            sequence_list.append(sequence)
        sequences = tf.ragged.constant(sequence_list)
        ds = tf.data.Dataset.from_tensor_slices((subject_list, sequences))

        dataloader = BERT4RecDataloader(max_seq_len=seq_max_len, max_predictions_per_seq=5)

        dataloader.generate_vocab(sequences)
        prepared_ds = dataloader.preprocess_dataset(ds, finetuning=True)
        return prepared_ds, dataloader

    def test_evaluate_batch(self):
        ds_size = 1000
        max_seq_len = 10

        prepared_ds, dataloader = self._create_test_dataset(ds_size=ds_size, seq_max_len=max_seq_len)
        prepared_batches = dataloader_utils.make_batches(prepared_ds, batch_size=5)
        evaluator = evaluation.get()

        # load a specific config
        config_path = pathlib.Path("../../config/bert_train_configs/ml-1m_128.json")
        config = utils.load_json_config(config_path)

        bert_encoder = networks.BertEncoder(dataloader.tokenizer.get_vocab_size(), **config)
        model = BERTModel(bert_encoder)
        model_wrapper = BERT4RecModelWrapper(model)
        # makes sure the weights are built
        _ = model(model.inputs)

        # usage of set in between to eliminate duplicates
        random_popular_items = list(set([random.randint(0, dataloader.tokenizer.get_vocab_size())
                                         for _ in range(200)]))

        metrics = evaluator.evaluate(model_wrapper, prepared_batches, popular_items_ranking=random_popular_items)

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


if __name__ == '__main__':
    tf.test.main()
