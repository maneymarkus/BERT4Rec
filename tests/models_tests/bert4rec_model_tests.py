from absl import logging
import random
import tensorflow as tf

from bert4rec.dataloaders import BERT4RecDataloader, dataloader_utils
from bert4rec.models.bert4rec_model import BERT4RecModel
from bert4rec.models.components import networks
from bert4rec.trainers import optimizers
import tests.test_utils as test_utils


class BERT4RecModelTests(tf.test.TestCase):
    # Warning: May take some time
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.optimizer = optimizers.get()

    def tearDown(self):
        super().tearDown()
        self.optimizer = None

    def _build_model(self,
                     vocab_size: int,
                     hidden_size: int,
                     num_layers: int,
                     optimizer: tf.keras.optimizers.Optimizer = None):
        bert_encoder = networks.Bert4RecEncoder(vocab_size=vocab_size,
                                                hidden_size=hidden_size,
                                                num_layers=num_layers)
        bert4rec_model = BERT4RecModel(bert_encoder)
        if optimizer is None:
            optimizer = self.optimizer
        bert4rec_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        # makes sure the weights are built
        _ = bert4rec_model(bert4rec_model.inputs)
        return bert4rec_model

    def test_call_model(self):
        ds_size = 1000
        vocab_size = 200
        batch_size = 64
        hidden_size = 768
        num_layers = 4
        max_seq_length = 100
        max_predictions = 5

        dataloader = BERT4RecDataloader(max_seq_len=max_seq_length, max_predictions_per_seq=max_predictions)
        ds = test_utils.generate_random_sequence_dataset(ds_size, vocab_size=vocab_size)
        dataloader.generate_vocab(ds)
        ds = dataloader.process_data(ds, finetuning=True)

        # overwrite vocab size, as dataloader might add special tokens to vocab size
        vocab_size = dataloader.get_tokenizer().get_vocab_size()
        model = self._build_model(vocab_size, hidden_size, num_layers)

        ds_batches = dataloader_utils.make_batches(ds, batch_size=batch_size)

        model_input = None
        for batch in ds_batches.take(1):
            model_input = batch

        # with masked language model
        output = model(model_input)
        self.assertIsInstance(output, dict)
        output_keys = {"sequence_output", "pooled_output", "encoder_outputs", "mlm_logits"}
        self.assertAllInSet(list(output.keys()), output_keys)

        sequence_output = output["sequence_output"]
        expected_sequence_output_shape = (batch_size, max_seq_length, hidden_size)
        self.assertEqual(sequence_output.shape, expected_sequence_output_shape)

        pooled_output = output["pooled_output"]
        expected_pooled_output_shape = (batch_size, hidden_size)
        self.assertEqual(pooled_output.shape, expected_pooled_output_shape)

        encoder_outputs = output["encoder_outputs"]
        self.assertLen(encoder_outputs, num_layers)
        encoder_output = encoder_outputs[0]
        expected_encoder_output_shape = (batch_size, max_seq_length, hidden_size)
        self.assertEqual(encoder_output.shape, expected_encoder_output_shape)

        mlm_logits = output["mlm_logits"]
        expected_mlm_logits_shape = (batch_size, max_predictions, vocab_size)
        self.assertEqual(mlm_logits.shape, expected_mlm_logits_shape)

        # without masked language model
        del model_input["masked_lm_positions"]
        output = model(model_input)
        self.assertIsInstance(output, dict)
        output_keys = {"sequence_output", "pooled_output", "encoder_outputs"}
        self.assertAllInSet(list(output.keys()), output_keys)

    def test_rank_items(self):
        ds_size = 1
        hidden_size = 768
        num_layers = 4
        max_sequence_length = 50

        dataloader = BERT4RecDataloader(max_seq_len=max_sequence_length, max_predictions_per_seq=5)
        ds = test_utils.generate_random_sequence_dataset(ds_size, max_seq_len=max_sequence_length)
        dataloader.generate_vocab(ds)
        ds = dataloader.process_data(ds, finetuning=True)
        batches = dataloader_utils.make_batches(ds)
        vocab_size = dataloader.tokenizer.get_vocab_size()

        bert4rec_model = self._build_model(vocab_size, hidden_size=hidden_size, num_layers=num_layers)

        encoder_input = None
        for el in batches.take(1):
            encoder_input = el

        # use individual ranking lists for each token -> shape of rank_items list: (batch, tokens, rank_items)
        rank_items = [
            [
                [random.randint(0, vocab_size - 1) for _ in range(random.randint(3, 10))]
                for _ in range(len(encoder_input["masked_lm_positions"][0]))
            ]
        ]

        rankings = bert4rec_model.rank_items(encoder_input,
                                             rank_items)

        # iterate over batches
        for b in range(len(rankings)):
            self.assertLessEqual(len(rankings[b]), len(encoder_input["masked_lm_positions"][b]),
                                 f"The amount of individual rankings per batch should be less than or "
                                 f"equal to the amount of masked_lm_positions "
                                 f"({len(encoder_input['masked_lm_positions'][b])}) "
                                 f"but received {len(rankings[b])} individual rankings.")
            # iterate over tokens
            for i, token_idx in enumerate(rankings[b]):
                self.assertEqual(len(rankings[b][i]), len(rank_items[b][i]),
                                 f"The length of an individual ranking list ({len(rankings[b][i])}) should be "
                                 f"equal to the corresponding original (unranked) items list ("
                                 f"{len(rank_items[b][i])})")


if __name__ == "__main__":
    tf.test.main()
