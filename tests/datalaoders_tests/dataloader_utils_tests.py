from absl import logging
import pandas as pd
import random
import string
import tensorflow as tf
import tensorflow_text as tf_text
import unittest

import bert4rec.dataloaders.dataloader_utils as utils


class DataloaderUtilsTest(unittest.TestCase):
    def test_convert_df_into_ds(self):
        logging.set_verbosity(logging.DEBUG)
        ds_size = 5
        test_list = [
            [
                random.randint(0, 9),
                list(random.choice(string.ascii_letters) for _ in range(random.randint(5, 15)))
            ] for _ in range(ds_size)
        ]
        df = pd.DataFrame(test_list, columns=["column_1", "column_2"])
        datatypes = [None, "list"]
        logging.debug(df)
        ds = utils.convert_df_to_ds(df, datatypes)
        self.assertIsInstance(ds, tf.data.Dataset,
                              f"The returned object should be of type tf.data.Dataset, "
                              f"but is of type {type(ds)}")

        actual_ds_size = tf.data.experimental.cardinality(ds)
        if actual_ds_size is not tf.data.experimental.UNKNOWN_CARDINALITY:
            self.assertEqual(actual_ds_size, ds_size,
                             f"The size of the converted dataset should be equal to its expected size ({ds_size}), "
                             f"but actually is {actual_ds_size}")

        for tensor1, tensor2 in ds.skip(2).take(1):
            self.assertIsInstance(tensor1, tf.Tensor,
                                  f"The first element of a dataset tuple should be of type tf.Tensor, "
                                  f"but is of type {type(tensor1)}")
            self.assertIsInstance(tensor2, tf.Tensor,
                                  f"The second element of a dataset tuple should be of type tf.Tensor, "
                                  f"but is of type {type(tensor2)}")
            self.assertEqual(test_list[2][0], tensor1.numpy(),
                             f"The (first) tensor value should be equal to it's 'generator' value "
                             f"({test_list[2][0]}), but has a value of {tensor1.numpy()}")
            tensor2_value = [b.decode("utf-8") for b in tensor2.numpy().tolist()]
            self.assertListEqual(test_list[2][1], tensor2_value,
                                 f"The (second) tensor value should be equal to it's 'generator' value "
                                 f"({test_list[2][1]}), but has a value of {tensor2_value}")
        logging.debug(ds)

    def test_masking_task(self):
        logging.set_verbosity(logging.DEBUG)
        vocab_size = 1000
        seq_length = 100
        values = list(random.randint(0, vocab_size - 1) for _ in range(seq_length))
        logging.debug(values)
        original_tensor = tf.ragged.constant([values])
        # max selections per batch
        mspb = 25
        # start token id
        sti = 0
        # end token id
        eti = 1
        # unknown token id
        uti = 4
        # mask token id
        mti = 2
        # selection rate
        sr = 0.15
        # mask token rate
        mtr = 0.8
        # random token rate
        rtr = 0.1
        random_selector, mask_values_chooser, masked_token_ids, masked_lm_positions, masked_lm_ids = \
            utils.apply_dynamic_masking_task(
                segments=original_tensor,
                max_selections_per_batch=mspb,
                mask_token_id=mti,
                special_token_ids=[sti, eti, uti],
                vocab_size=vocab_size,
                selection_rate=sr,
                mask_token_rate=mtr,
                random_token_rate=rtr
            )

        self.assertIsInstance(random_selector, tf_text.RandomItemSelector,
                              f"The first return value of the util function that applies the dynamic language model "
                              f"should be a tensorflow_text.RandomItemSelector, "
                              f"but is of type {type(random_selector)}")

        self.assertIsInstance(mask_values_chooser, tf_text.MaskValuesChooser,
                              f"Th second return value of the util function that applies the dynamic language model "
                              f"should be a tensorflow_text.MaskValuesChooser, "
                              f"but is of type {type(mask_values_chooser)}")

        len_original_tensor = len(original_tensor.numpy()[0])
        len_masked_tensor = len(masked_token_ids.numpy()[0])
        self.assertEqual(len_original_tensor, len_masked_tensor,
                         f"The length of the masked tensor should be the exact same length as the original tensor "
                         f"(which is: {len_original_tensor}), but has a length of: "
                         f"{len_masked_tensor}")

        number_masked_token_positions = len(masked_lm_positions.numpy()[0])
        self.assertLessEqual(number_masked_token_positions, 25,
                             f"The number of masked tokens should be less than or equal to 25, "
                             f"but is: {number_masked_token_positions}")
        self.assertGreater(number_masked_token_positions, 0,
                           f"The number of masked tokens should be greater than 0, "
                           f"but is {number_masked_token_positions}")

        number_masked_tokens = len(masked_lm_ids.numpy()[0])
        self.assertEqual(number_masked_tokens, number_masked_token_positions,
                         f"The number of masked token positions ({number_masked_token_positions}) "
                         f"and the number of masked tokens ({number_masked_tokens}) should be equal.")

        masked_tokens = masked_lm_ids.numpy()[0]
        self.assertTrue(all(x in values for x in masked_tokens),
                        f"Every token that got masked should appear in the original values list")

        masked_token_positions = masked_lm_positions.numpy()[0]
        self.assertTrue(all(0 <= x <= seq_length for x in masked_token_positions),
                        f"Each position of every masked token should be greater than or equal to 0 and "
                        f"less than or equal to the sequence length")

    def test_split_dataset(self):
        logging.set_verbosity(logging.DEBUG)
        ds_size = 5000
        train_split = 0.7
        val_split = 0.2
        test_split = 0.1
        test_list = [
            [
                random.randint(0, 9),
                random.randint(0, 9)
            ] for _ in range(ds_size)
        ]
        logging.debug(test_list)
        df = pd.DataFrame(test_list, columns=["column_1", "column_2"])
        logging.debug(df)
        ds = utils.convert_df_to_ds(df)
        logging.debug(ds)

        with self.assertRaises(ValueError):
            _ = utils.split_dataset(ds,
                                    train_split=0.8,
                                    val_split=0.2,
                                    test_split=0.2)

        expected_train_ds_size = ds_size * train_split
        expected_val_ds_size = ds_size * val_split
        expected_test_ds_size = ds_size * test_split
        train_ds, val_ds, test_ds = utils.split_dataset(ds,
                                                        train_split=train_split,
                                                        val_split=val_split,
                                                        test_split=test_split)

        actual_train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        actual_val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
        actual_test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()

        if actual_train_ds_size is not tf.data.experimental.UNKNOWN_CARDINALITY:
            self.assertEqual(actual_train_ds_size, expected_train_ds_size,
                             f"The split training dataset should have an expected size of {expected_train_ds_size}, "
                             f"but actually has a size of {actual_train_ds_size}")
        if actual_val_ds_size is not tf.data.experimental.UNKNOWN_CARDINALITY:
            self.assertEqual(actual_val_ds_size, expected_val_ds_size,
                             f"The split training dataset should have an expected size of {expected_val_ds_size}, "
                             f"but actually has a size of {actual_val_ds_size}")
        if actual_test_ds_size is not tf.data.experimental.UNKNOWN_CARDINALITY:
            self.assertEqual(actual_test_ds_size, expected_test_ds_size,
                             f"The split training dataset should have an expected size of {expected_test_ds_size}, "
                             f"but actually has a size of {actual_test_ds_size}")
