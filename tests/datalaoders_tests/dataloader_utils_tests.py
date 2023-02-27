from absl import logging
import numpy as np
import pandas as pd
import random
import string
import tensorflow as tf

import bert4rec.dataloaders.dataloader_utils as utils


class DataloaderUtilsTest(tf.test.TestCase):
    def setUp(self):
        super(DataloaderUtilsTest, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        pass

    def test_rank_items_by_popularity(self):
        list_1 = [0, 5, 2, 8, 4, 6, 3, 6, 2, 7, 5, 8, 9, 4, 5, 6, 6, 6, 6, 6, 1, 1, 3, 7, 7, 9, 9, 9]
        sorted_list_1 = utils.rank_items_by_popularity(list_1)
        self.assertEqual(len(sorted_list_1), len(set(list_1)),
                         f"The ranked list should have no duplicates and therefore should have as many elements as a"
                         f"set generated from that list ({len(set(list_1))}) but actually has a length of: "
                         f"{len(sorted_list_1)}")
        self.assertEqual(sorted_list_1[0], 6,
                         f"The most popular number in this list: \n{list_1}\nshould be 6 but is: {sorted_list_1[0]}")
        self.assertEqual(sorted_list_1[9], 0,
                         f"The least popular number in this list: \n{list_1}\nshould be 0 but is: {sorted_list_1[9]}")

    def test_split_sequence_df(self):
        df_size = 6
        subject_list = []
        item_list = []
        for _ in range(df_size):
            subject = random.randint(0, 100)
            # length of sequence
            for _ in range(random.randint(10, 50)):
                subject_list.append(subject)
                item = random.randint(0, 1000)
                item_list.append(item)

        # add special case 1
        special_subject_1 = random.randint(100, 200)
        subject_list.append(special_subject_1)
        item_list.append(random.randint(0, 1000))

        # add special case 2
        special_subject_2 = random.randint(100, 200)
        for _ in range(2):
            subject_list.append(special_subject_2)
            item_list.append(random.randint(0, 1000))

        # add special case 3
        special_subject_3 = random.randint(100, 200)
        for _ in range(3):
            subject_list.append(special_subject_3)
            item_list.append(random.randint(0, 1000))

        data = {
            "subject": subject_list,
            "item": item_list
        }

        group_by_column = "subject"
        sequence_column = "item"

        df = pd.DataFrame(data)

        train_df, val_df, test_df = utils.split_sequence_df(
            df,
            group_by_column,
            [sequence_column],
            min_sequence_length=3
        )
        # 3 special cases
        expected_df_size = df_size + 3
        self.assertEqual(train_df.shape[0], expected_df_size,
                         f"The generated train dataframe should have {expected_df_size} number of rows "
                         f"(one special case should be filtered out) but the actual number of rows is: "
                         f"{train_df.shape[0]} ")
        self.assertLessEqual(val_df.shape[0], train_df.shape[0],
                             f"The generated validation dataframe should have less or an equal amount of rows "
                             f"compared to the generated train dataframe ({train_df.shape[0]}) but actually has: "
                             f"{val_df.shape[0]}")
        self.assertLessEqual(test_df.shape[0], val_df.shape[0],
                             f"The generated test dataframe should have less or an equal amount of rows "
                             f"compared to the generated validation dataframe ({val_df.shape[0]}) but actually has: "
                             f"{test_df.shape[0]}")
        self.assertEqual(val_df.shape[0], expected_df_size - 2,
                         f"The generated validation dataframe should have {expected_df_size - 2} number of rows "
                         f"(two special cases should be filtered out) but the actual number of rows is: "
                         f"{val_df.shape[0]}")
        self.assertEqual(test_df.shape[0], expected_df_size - 2,
                         f"The generated test dataframe should have {expected_df_size - 2} number of rows "
                         f"(two special cases should be filtered out) but the actual number of rows is: "
                         f"{test_df.shape[0]}")
        self.assertEqual(len(val_df["item"].iloc[-1]), 2,
                         f"The last row of the generated validation dataframe should contain a sequence of length 2 "
                         f"but actually has: {len(val_df['item'].iloc[-1])}.\nSequence:{val_df['item'].iloc[-1]}")
        self.assertEqual(len(test_df["item"].iloc[-1]), 3,
                         f"The last row of the generated test dataframe should contain a sequence of length 3 "
                         f"but actually has: {len(test_df['item'].iloc[-1])}.\nSequence:{test_df['item'].iloc[-1]}")

    def test_convert_df_into_ds(self):
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

    def test_make_sequence_df(self):
        df_size = 100
        test_list = [
            [
                random.randint(0, 9),
                random.randint(0, 9),
                random.choice(string.ascii_letters)
            ] for _ in range(df_size)
        ]
        df = pd.DataFrame(test_list, columns=["column_1", "column_2", "column_3"])
        sequence_df_1 = utils.make_sequence_df(df, "column_1", ["column_2"])
        self.assertEqual(len(sequence_df_1.columns), 1)
        self.assertEqual(sequence_df_1.columns.tolist(), ["column_2"])
        sequence_df_2 = utils.make_sequence_df(df, "column_1", ["column_2", "column_3"])
        self.assertEqual(len(sequence_df_2.columns), 2)
        self.assertEqual(sequence_df_2.columns.tolist(), ["column_2", "column_3"])
        for i in sequence_df_1.index:
            self.assertIsInstance(sequence_df_1["column_2"][i], list)
        for i in sequence_df_2.index:
            self.assertIsInstance(sequence_df_2["column_2"][i], list)
            self.assertIsInstance(sequence_df_2["column_3"][i], list)

    def test_duplicate_dataset(self):
        ds_values = [random.randint(0, 9) for _ in range(random.randint(20, 30))]
        ds = tf.data.Dataset.from_tensor_slices(ds_values)
        ds_size = ds.cardinality()
        duplication_factor = random.randint(2, 6)
        duplicated_ds = utils.duplicate_dataset(ds, duplication_factor)
        expected_ds_size = ds_size * duplication_factor
        self.assertEqual(duplicated_ds.cardinality(), expected_ds_size)
        non_duplicated_ds = utils.duplicate_dataset(ds, 1)
        self.assertEqual(non_duplicated_ds.cardinality(), ds_size)
        with self.assertRaises(ValueError):
            utils.duplicate_dataset(ds, 0)
            utils.duplicate_dataset(ds, -1)
            utils.duplicate_dataset(ds, -6)

    def test_masking_task(self):
        vocab_size = 1000
        seq_length = 100
        original_values = np.array([random.randint(0, vocab_size - 1) for _ in range(seq_length)])
        logging.debug(original_values)
        # max selections per sequence
        msps = 25
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
        masked_token_ids, masked_lm_positions, masked_lm_ids = \
            utils.apply_dynamic_masking_task(
                sequence=original_values,
                max_selections_per_seq=msps,
                mask_token_id=mti,
                special_token_ids=[sti, eti, uti],
                vocab_size=vocab_size,
                selection_rate=sr,
                mask_token_rate=mtr,
                random_token_rate=rtr
            )

        len_original_values = len(original_values)
        len_masked_tensor = len(masked_token_ids)
        self.assertEqual(len_original_values, len_masked_tensor,
                         f"The length of the masked values array should be the exact same length as "
                         f"the original tensor (which is: {len_original_values}), but has a "
                         f"length of: {len_masked_tensor}")

        number_masked_token_positions = len(masked_lm_positions)
        self.assertLessEqual(number_masked_token_positions, 25,
                             f"The number of masked tokens should be less than or equal to 25, "
                             f"but is: {number_masked_token_positions}")
        self.assertGreater(number_masked_token_positions, 0,
                           f"The number of masked tokens should be greater than 0, "
                           f"but is {number_masked_token_positions}")

        number_masked_tokens = len(masked_lm_ids)
        self.assertEqual(number_masked_tokens, number_masked_token_positions,
                         f"The number of masked token positions ({number_masked_token_positions}) "
                         f"and the number of masked tokens ({number_masked_tokens}) should be equal.")

        masked_tokens = masked_lm_ids
        self.assertTrue(all(x in original_values for x in masked_tokens),
                        f"Every token that got masked should appear in the original values list")

        masked_token_positions = masked_lm_positions
        self.assertTrue(all(0 <= x <= seq_length for x in masked_token_positions),
                        f"Each position of every masked token should be greater than or equal to 0 and "
                        f"less than or equal to the sequence length")

    def test_mask_last_token_only(self):
        mask_token_id = 1
        values = np.array([random.randint(10, 50) for _ in range(random.randint(20, 30))])
        masked_tensor, masked_positions, masked_tokens = \
            utils.mask_last_token_only(values, mask_token_id)
        last_value = masked_tensor[len(masked_tensor) - 1]
        self.assertEqual(last_value, mask_token_id)

    def test_split_dataset(self):
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


if __name__ == "__main__":
    tf.test.main()
