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
import tensorflow as tf

from bert4rec import dataloaders
from tests import test_utils


class BERT4RecDataloaderTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.dataloader: dataloaders.BaseDataloader = None
        self.expected_vocab_size = None

    def tearDown(self):
        self.dataloader = None

    def test_dataset_identifier(self):
        if self.dataloader is None:
            return

        id = self.dataloader.dataset_identifier
        self.assertIsInstance(id, str)

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

    def test_get_data(self):
        if self.dataloader is None:
            return

        ds = self.dataloader.get_data(split_data=False)[0]
        self.assertIsInstance(ds, tf.data.Dataset)

        train_ds, val_ds, test_ds = self.dataloader.get_data(split_data=True, apply_mlm=False)
        self.assertIsInstance(train_ds, tf.data.Dataset)
        self.assertIsInstance(val_ds, tf.data.Dataset)
        self.assertIsInstance(test_ds, tf.data.Dataset)

        train_ds, val_ds, test_ds = self.dataloader.get_data(
            split_data=True, apply_mlm=True, finetuning_split=1
        )
        self.assertIsInstance(train_ds, tf.data.Dataset)
        self.assertIsInstance(val_ds, tf.data.Dataset)
        self.assertIsInstance(test_ds, tf.data.Dataset)

        with self.assertRaises(ValueError):
            self.dataloader.get_data(finetuning_split=1.1)
            self.dataloader.get_data(finetuning_split=-0.1)
            self.dataloader.get_data(duplication_factor=0)
            self.dataloader.get_data(duplication_factor=-1)

    def test_load_data(self):
        if self.dataloader is None:
            return

        ds = self.dataloader.load_data(split_data=False)[0]
        self.assertIsInstance(ds, tf.data.Dataset)

        train_ds, val_ds, test_ds = self.dataloader.load_data(split_data=True)
        self.assertIsInstance(train_ds, tf.data.Dataset)
        self.assertIsInstance(val_ds, tf.data.Dataset)
        self.assertIsInstance(test_ds, tf.data.Dataset)

        with self.assertRaises(ValueError):
            self.dataloader.load_data(duplication_factor=0)
            self.dataloader.load_data(duplication_factor=-1)
            self.dataloader.load_data(extract_data=["one", "two"], datatypes=["three"])
            self.dataloader.load_data(extract_data=["one"], datatypes=["two", "three"])

    def test_process_data(self):
        if self.dataloader is None:
            return

        ds_size = 100
        seed = 5

        ds = test_utils.generate_random_sequence_dataset(ds_size=ds_size, seed=seed)
        # no mlm
        ds_1 = self.dataloader.process_data(ds, apply_mlm=False)
        self.assertIsInstance(ds_1, tf.data.Dataset)

        # mlm, no finetuning
        ds_2 = self.dataloader.process_data(ds, apply_mlm=True, finetuning=False)
        self.assertIsInstance(ds_2, tf.data.Dataset)

        # mlm, finetuning
        ds_3 = self.dataloader.process_data(ds, apply_mlm=True, finetuning=True)
        self.assertIsInstance(ds_3, tf.data.Dataset)

    def test_prepare_training(self):
        if self.dataloader is None:
            return

        # method should work fine without any arguments
        train_ds, val_ds, test_ds = self.dataloader.prepare_training()
        self.assertIsInstance(train_ds, tf.data.Dataset)
        self.assertIsInstance(val_ds, tf.data.Dataset)
        self.assertIsInstance(test_ds, tf.data.Dataset)

    def test_prepare_inference(self):
        if self.dataloader is None:
            return

        seq_len = 45
        seed = 81

        sequence = test_utils.generate_random_word_list(size=seq_len, seed=seed)
        prepared_sequence = self.dataloader.prepare_inference(sequence)
        self.assertIsInstance(prepared_sequence, dict)
        dict_keys = set(prepared_sequence.keys())
        self.assertContainsSubset(
            {"input_word_ids", "input_mask"},
            dict_keys
        )

    def test_create_item_list(self):
        if self.dataloader is None:
            return

        item_list = self.dataloader.create_item_list()
        self.assertIsInstance(item_list, list)

    def test_create_item_list_tokenized(self):
        if self.dataloader is None:
            return

        tokenized_item_list = self.dataloader.create_item_list_tokenized()
        self.assertIsInstance(tokenized_item_list, list)

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


class BERT4RecML1MDataloaderTests(BERT4RecDataloaderTests):
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecML1MDataloader()
        self.expected_vocab_size = 3706 + len(self.dataloader._SPECIAL_TOKENS)

    def test_dataset_identifier(self):
        super().test_dataset_identifier()

    def test_generate_vocab(self):
        super().test_generate_vocab()

    def test_get_data(self):
        super().test_get_data()

    def test_load_data(self):
        super().test_load_data()

    def test_process_data(self):
        super().test_process_data()

    def test_prepare_training(self):
        super().test_prepare_training()

    def test_prepare_inference(self):
        super().test_prepare_inference()

    def test_create_item_list(self):
        super().test_create_item_list()

    def test_create_item_list_tokenized(self):
        super().test_create_item_list_tokenized()

    def test_create_popular_item_ranking(self):
        super().test_create_popular_item_ranking()

    def test_create_popular_item_ranking_tokenized(self):
        super().test_create_popular_item_ranking_tokenized()


class BERT4RecML20MDataloaderTests(BERT4RecDataloaderTests):
    """
    NOTE: Be aware that especially these tests require a lot of computation time and power
    as this dataset contains about 139.000 rather long sequences
    """
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecML20MDataloader()
        self.expected_vocab_size = 26729 + len(self.dataloader._SPECIAL_TOKENS)


class BERT4RecRedditDataloaderTests(BERT4RecDataloaderTests):
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecRedditDataloader()
        self.expected_vocab_size = 335420 + len(self.dataloader._SPECIAL_TOKENS)


class BERT4RecBeautyDataloaderTests(BERT4RecDataloaderTests):
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecBeautyDataloader()
        self.expected_vocab_size = 54542 + len(self.dataloader._SPECIAL_TOKENS)


class BERT4RecSteamDataloaderTests(BERT4RecDataloaderTests):
    """
    NOTE: Be aware that especially these tests require a lot of computation time and power
    as this dataset contains about 281,000 sequences
    """
    def setUp(self):
        super().setUp()
        self.dataloader = dataloaders.BERT4RecSteamDataloader()
        self.expected_vocab_size = 13044 + len(self.dataloader._SPECIAL_TOKENS)


if __name__ == '__main__':
    tf.test.main()
