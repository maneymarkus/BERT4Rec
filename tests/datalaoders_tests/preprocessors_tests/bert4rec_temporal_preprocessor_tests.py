from absl import logging
import random
import string
import tensorflow as tf

from bert4rec import tokenizers
from bert4rec.dataloaders import preprocessors
from tests import test_utils


class BERT4RecTemporalPreprocessorsTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.preprocessor = preprocessors.BERT4RecTemporalPreprocessor
        self.tokenizer = tokenizers.SimpleTokenizer()
        self.max_seq_len = 100
        self.max_predictions_per_seq = 25
        self.mask_token_id = 0
        self.unk_token_id = 1
        self.pad_token_id = 2
        self.masked_lm_rate = 0.25
        self.mask_token_rate = 0.8
        self.random_token_rate = 0.1

    def tearDown(self):
        pass

    def _initiate_preprocessor(self):
        # block these special tokens in the tokenizer
        self.tokenizer.tokenize([
            "[MASK]",
            "[UNK]",
            "[PAD]"
        ])
        self.preprocessor.set_properties(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            max_predictions_per_seq=self.max_predictions_per_seq,
            mask_token_id=self.mask_token_id,
            unk_token_id=self.unk_token_id,
            pad_token_id=self.pad_token_id,
            masked_lm_rate=self.masked_lm_rate,
            mask_token_rate=self.mask_token_rate,
            random_token_rate=self.random_token_rate
        )

    def test_set_properties(self):
        self.preprocessor.set_properties(tokenizer=self.tokenizer)
        self.assertEqual(type(self.preprocessor.tokenizer), tokenizers.SimpleTokenizer)
        self.assertIsNotNone(self.preprocessor.tokenizer)
        self.preprocessor.set_properties(pad_token_id=5, mask_token_rate=0.3)
        self.assertEqual(type(self.preprocessor.tokenizer), tokenizers.SimpleTokenizer)
        self.assertEqual(self.preprocessor.pad_token_id, 5)
        self.assertEqual(self.preprocessor.mask_token_rate, 0.3)
        self.preprocessor.set_properties(pad_token_id=0, unk_token_id=3)
        self.assertEqual(self.preprocessor.pad_token_id, 0)
        self.assertEqual(self.preprocessor.unk_token_id, 3)

    def test_process_element(self):
        self._initiate_preprocessor()
        sequence = [random.choice(string.ascii_letters) for _ in range(25)]
        sequence_tensor = tf.constant(sequence)
        timestamps = list(range(10000000, 10200000, 8000))
        timestamps_tensor = tf.constant(timestamps)
        # No Masked Language Model
        model_inputs = self.preprocessor.process_element(sequence_tensor,
                                                         timestamps_tensor,
                                                         apply_mlm=False,
                                                         finetuning=False)
        self.assertIsInstance(model_inputs, dict)
        self.assertEqual(len(list(model_inputs.values())), 4)
        for key, value in model_inputs.items():
            self.assertTrue(tf.is_tensor(value))
            self.assertEqual(len(value), self.max_seq_len)
            self.assertNotIn(self.mask_token_id, value)
            self.assertIn(self.pad_token_id, value)
            self.assertEqual(value[len(sequence)], self.pad_token_id)
            self.assertEqual(value[len(value) - 1], self.pad_token_id)

        # Masked Language Model, No Finetuning
        model_inputs_2 = self.preprocessor.process_element(sequence_tensor,
                                                           timestamps_tensor,
                                                           apply_mlm=True,
                                                           finetuning=False)
        self.assertIsInstance(model_inputs_2, dict)
        self.assertEqual(len(list(model_inputs_2.values())), 7)
        sub_dict_1 = {
            k: model_inputs_2[k] for k in [
                "labels",
                "input_word_ids",
                "input_mask",
                "input_timestamps"
            ] if k in model_inputs_2
        }
        sub_dict_2 = {
            k: model_inputs_2[k] for k in [
                "masked_lm_ids",
                "masked_lm_positions",
                "masked_lm_weights"
            ] if k in model_inputs_2
        }
        self.assertIn(self.mask_token_id, model_inputs_2["input_word_ids"])
        for key, value in sub_dict_1.items():
            self.assertTrue(tf.is_tensor(value))
            self.assertEqual(len(value), self.max_seq_len)
            self.assertIn(self.pad_token_id, value)
            self.assertEqual(value[len(sequence)], self.pad_token_id)
            self.assertEqual(value[len(value) - 1], self.pad_token_id)

        for key, value in sub_dict_2.items():
            self.assertTrue(tf.is_tensor(value))
            self.assertEqual(len(value), self.max_predictions_per_seq)
            self.assertIn(self.pad_token_id, value)

        # Masked Language Model, Finetuning
        model_inputs_3 = self.preprocessor.process_element(sequence_tensor,
                                                           timestamps_tensor,
                                                           apply_mlm=True,
                                                           finetuning=True)
        self.assertIsInstance(model_inputs_3, dict)
        self.assertEqual(len(list(model_inputs_3.values())), 7)
        sub_dict_1 = {
            k: model_inputs_3[k] for k in [
                "labels",
                "input_word_ids",
                "input_mask",
                "input_timestamps"
            ] if k in model_inputs_3
        }
        sub_dict_2 = {
            k: model_inputs_3[k] for k in [
                "masked_lm_ids",
                "masked_lm_positions",
                "masked_lm_weights"
            ] if k in model_inputs_3
        }
        self.assertEqual(model_inputs_3["input_word_ids"][len(sequence) - 1], self.mask_token_id)
        for key, value in sub_dict_1.items():
            self.assertTrue(tf.is_tensor(value))
            self.assertEqual(len(value), self.max_seq_len)
            self.assertIn(self.pad_token_id, value)
            self.assertEqual(value[len(sequence)], self.pad_token_id)
            self.assertEqual(value[len(value) - 1], self.pad_token_id)

        for key, value in sub_dict_2.items():
            self.assertTrue(tf.is_tensor(value))
            self.assertEqual(len(value), self.max_predictions_per_seq)
            self.assertIn(self.pad_token_id, value)
            self.assertNotEqual(value[0], self.pad_token_id)
            self.assertEqual(value[1], self.pad_token_id)

        # Oversize
        oversize_sequence = [
            random.choice(string.ascii_letters) for _ in range(self.max_seq_len + 20)
        ]
        oversize_seq_tensor = tf.constant(oversize_sequence)
        oversize_timestamps = list(range(10000000, 10200000, 1665))
        oversize_time_tensor = tf.constant(oversize_timestamps)
        model_inputs_4 = self.preprocessor.process_element(oversize_seq_tensor,
                                                           oversize_time_tensor,
                                                           apply_mlm=True,
                                                           finetuning=False)
        self.assertIsInstance(model_inputs_4, dict)
        self.assertEqual(len(list(model_inputs_4.values())), 7)
        sub_dict_1 = {
            k: model_inputs_4[k] for k in [
                "labels",
                "input_word_ids",
                "input_mask",
                "input_timestamps"
            ] if k in model_inputs_4
        }
        for key, value in sub_dict_1.items():
            self.assertTrue(tf.is_tensor(value))
            self.assertEqual(len(value), self.max_seq_len)

    def test_process_dataset(self):
        self._initiate_preprocessor()
        ds_size = 100
        seed = 1234
        timestamp_start = 10000000
        print(tf.executing_eagerly())

        def add_timestamps(seq):
            timestamps = [
                timestamp_start + random.randint(10000, 20000) * 3 * i for i in range(len(seq))
            ]
            return seq, timestamps

        ds = test_utils.generate_random_sequence_dataset(ds_size=ds_size, seed=seed)
        ds = ds.map(lambda seq: tf.py_function(
            func=add_timestamps,
            inp=[seq],
            Tout=[tf.string, tf.int32]
        ))
        prepared_ds_without_mlm = self.preprocessor.process_dataset(ds, apply_mlm=False, finetuning=False)
        self.assertIsInstance(prepared_ds_without_mlm, tf.data.Dataset)
        for el in prepared_ds_without_mlm:
            self.assertEqual(type(el), dict)
            self.assertEqual(len(list(el.values())), 4)

        prepared_ds_with_mlm = self.preprocessor.process_dataset(ds, apply_mlm=True, finetuning=False)
        self.assertIsInstance(prepared_ds_with_mlm, tf.data.Dataset)
        for el in prepared_ds_with_mlm:
            self.assertEqual(type(el), dict)
            self.assertEqual(len(list(el.values())), 7)

    def test_prepare_inference(self):
        self._initiate_preprocessor()
        sequence = [random.choice(string.ascii_letters) for _ in range(25)]
        timestamps = list(range(10000000, 10200000, 8000))
        data = (sequence, timestamps)
        prepared_inference = self.preprocessor.prepare_inference(data)
        self.assertIsInstance(prepared_inference, dict)
        self.assertEqual(len(list(prepared_inference.values())), 7)
        self.assertEqual(
            len(prepared_inference["input_word_ids"][0]),
            self.max_seq_len
        )
        self.assertEqual(
            prepared_inference["input_word_ids"][0][len(sequence)],
            self.mask_token_id
        )
        self.assertEqual(
            prepared_inference["input_word_ids"][0][len(sequence) + 1],
            self.pad_token_id
        )
        self.assertEqual(
            prepared_inference["masked_lm_ids"][0][1],
            self.pad_token_id
        )
        self.assertEqual(
            len(prepared_inference["input_timestamps"][0]),
            self.max_seq_len
        )


if __name__ == "__main__":
    tf.test.main()
