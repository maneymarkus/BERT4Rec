"""
This file tests all the available dataloaders for the BERT4Rec machine learning models.
There is a base test class providing the main (and actually only real) testing code since each
child class provides the same behaviour just with different data. Each child test class basically
just instantiates their respective dataloader. To be able to conveniently use features of the IDE
(e.g. PyCharm from JetBrains) that allow to execute single test functions from the editor each
child class implements every parent method and simply calls it. This "feature" may be omitted
and was just included for convenience reasons.

NOTE: Be aware that these tests require a lot of computation time and power. Especially the
bigger datasets (like ML-20M) require a lot of time
"""
from absl import logging
import numpy as np
import tensorflow as tf

from bert4rec import dataloaders
from tests import test_utils


class BERT4RecDataloaderTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.dataloader = None
        self.expected_vocab_size = None

    def tearDown(self):
        self.dataloader = None

    def _choose_random_element_from_dataset(self, ds: tf.data.Dataset, seed: int = 3):
        ds_item = ds.shuffle(1000, seed=seed).take(1)
        return list(ds_item.as_numpy_iterator())[0]

    def _assert_preprocessed_dataset(self, ds: tf.data.Dataset):
        # retrieve a single dataset element
        ds_item = self._choose_random_element_from_dataset(ds)

        self.assertIsInstance(ds_item, dict,
                              "A random item from a preprocessed dataset created by a BERT4rec dataloader "
                              f"method should be a dict, but is: {type(ds_item)}.")
        self.assertEqual(len(ds_item), 6,
                         "A random item from a preprocessed dataset BERT4Rec dataloader should be a dict with "
                         f"6 values, but this one actually has: {len(ds_item)} values.")
        dict_keys = ["masked_lm_ids", "masked_lm_positions", "masked_lm_weights", "labels",
                     "input_word_ids", "input_mask"]
        self.assertListEqual(list(ds_item.keys()), dict_keys)

    def test_load_data_into_ds(self, ds: tf.data.Dataset = None):
        if self.dataloader is None:
            return

        if ds is None:
            ds = self.dataloader.load_data_into_ds()
        self.assertIsInstance(ds, tf.data.Dataset)

        # retrieve a single dataset element
        ds_item = self._choose_random_element_from_dataset(ds)

        self.assertIsInstance(ds_item, tuple,
                              "A random item from a dataset created by a BERT4rec dataloader load_data_into...() "
                              f"method should be a tuple, but is: {type(ds_item)}.")
        self.assertEqual(len(ds_item), 2,
                         "A random item from a BERT4Rec dataloader load_data_init...() method should be a "
                         f"tuple with two values, but this one actually has: {len(ds_item)} values.")
        self.assertIsInstance(ds_item[0], np.int64,
                              "The first element from the dataset item tuple should be an np.int64, but is:"
                              f"{type(ds_item[0])}")
        self.assertIsInstance(ds_item[1], np.ndarray,
                              "The second element from the dataset item tuple should be an np.ndarray, but is:"
                              f"{type(ds_item[1])}")

    def test_load_data_into_split_ds(self):
        if self.dataloader is None:
            return

        train_ds, val_ds, test_ds = self.dataloader.load_data_into_split_ds()
        dss = [train_ds, val_ds, test_ds]
        for ds in dss:
            self.test_load_data_into_ds(ds)

    def test_generate_vocab(self):
        vocab_size = 200
        generic_dataloader = dataloaders.BERT4RecDataloader(0, 0)
        test_list = test_utils.generate_random_word_list(size=vocab_size)
        generic_dataloader.generate_vocab(test_list)
        special_tokens = generic_dataloader._SPECIAL_TOKENS
        tokenizer = generic_dataloader.get_tokenizer()
        expected_vocab_size = vocab_size + len(special_tokens)
        self.assertEqual(tokenizer.get_vocab_size(), expected_vocab_size)

        if self.dataloader is None:
            return

        self.dataloader.generate_vocab()
        self.assertEqual(self.dataloader.tokenizer.get_vocab_size(), self.expected_vocab_size)

    def test_create_popular_item_ranking(self):
        if self.dataloader is None:
            return

        pop_items_ranking = self.dataloader.create_popular_item_ranking()
        item_vocab_size = self.expected_vocab_size - len(self.dataloader._SPECIAL_TOKENS)
        self.assertIsInstance(pop_items_ranking, list)
        self.assertEqual(len(pop_items_ranking), item_vocab_size)

    def test_create_popular_item_ranking_tokenized(self):
        if self.dataloader is None:
            return

        tokenized_pop_items_ranking = self.dataloader.create_popular_item_ranking_tokenized()
        item_vocab_size = self.expected_vocab_size - len(self.dataloader._SPECIAL_TOKENS)
        self.assertIsInstance(tokenized_pop_items_ranking, list)
        self.assertEqual(len(tokenized_pop_items_ranking), item_vocab_size)

    def test_prepare_training(self):
        if self.dataloader is None:
            return

        train_ds, val_ds, test_ds = self.dataloader.prepare_training()
        dss = [train_ds, val_ds, test_ds]
        for ds in dss:
            self._assert_preprocessed_dataset(ds)

    def test_preprocess_dataset(self):
        if self.dataloader is None:
            return

        preprocessed_ds = self.dataloader.preprocess_dataset()
        self._assert_preprocessed_dataset(preprocessed_ds)
        preprocessed_ds_2 = self.dataloader.preprocess_dataset(apply_mlm=False)

        # retrieve a single dataset element
        ds_item = self._choose_random_element_from_dataset(preprocessed_ds_2)

        self.assertIsInstance(ds_item, dict,
                              "A random item from a dataset created by a BERT4rec dataloader "
                              f"prepare_training() method should be a dict, but is: {type(ds_item)}.")
        self.assertEqual(len(ds_item), 4,
                         "A random item from a BERT4Rec dataloader prepare_training() should be a dict with "
                         f"two values, but this one actually has: {len(ds_item)} values.")
        dict_keys = {"labels", "input_word_ids", "input_mask"}
        self.assertAllInSet(list(ds_item.keys()), dict_keys)

    def test_prepare_inference(self):
        if self.dataloader is None:
            return

        sequence = test_utils.generate_random_word_list()
        model_input = self.dataloader.prepare_inference(sequence)
        self.assertIsInstance(model_input, dict)
        dict_keys = {"labels", "input_word_ids", "input_mask",
                     "masked_lm_ids", "masked_lm_positions", "masked_lm_weights"}
        self.assertAllInSet(list(model_input.keys()), dict_keys)
        input_word_ids = model_input["input_word_ids"]
        self.assertEqual(tf.rank(input_word_ids).numpy(), 2)
        self.assertEqual(len(input_word_ids[0]), self.dataloader._MAX_SEQ_LENGTH)
        # convert input word ids to list and then remove all the "0" values
        input_word_ids = input_word_ids.numpy().tolist()[0]
        while 0 in input_word_ids:
            input_word_ids.remove(0)
        last_non_padding_item = input_word_ids[-1]
        self.assertEqual(last_non_padding_item, self.dataloader._MASK_TOKEN_ID)


class BERT4RecML1MDataloaderTests(BERT4RecDataloaderTests):
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecML1MDataloader()
        self.expected_vocab_size = 3710

    def test_load_data_into_ds(self, ds: tf.data.Dataset = None):
        super().test_load_data_into_ds(ds)

    def test_load_data_into_split_ds(self):
        super().test_load_data_into_split_ds()

    def test_generate_vocab(self):
        super().test_generate_vocab()

    def test_create_popular_item_ranking(self):
        super().test_create_popular_item_ranking()

    def test_create_popular_item_ranking_tokenized(self):
        super().test_create_popular_item_ranking_tokenized()

    def test_prepare_training(self):
        super().test_prepare_training()

    def test_preprocess_dataset(self):
        super().test_preprocess_dataset()

    def test_prepare_inference(self):
        super().test_prepare_inference()


class BERT4RecML20MDataloaderTests(BERT4RecDataloaderTests):
    """
    NOTE: Be aware that especially these tests require a lot of computation time and power
    as this dataset contains about 139.000 rather long sequences
    """
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecML20MDataloader()
        self.expected_vocab_size = 26733


"""
Not yet correctly implemented

class BERT4RecRedditDataloaderTests(BERT4RecDataloaderTests):
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecML1MDataloader()

    def test_load_data_into_ds(self, ds: tf.data.Dataset = None):
        super().test_load_data_into_ds(ds)
"""


class BERT4RecBeautyDataloaderTests(BERT4RecDataloaderTests):
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecBeautyDataloader()
        self.expected_vocab_size = 54546


class BERT4RecSteamDataloaderTests(BERT4RecDataloaderTests):
    """
    NOTE: Be aware that especially these tests require a lot of computation time and power
    as this dataset contains about 281,000 sequences
    """
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecSteamDataloader()
        self.expected_vocab_size = 13048


if __name__ == '__main__':
    tf.test.main()
