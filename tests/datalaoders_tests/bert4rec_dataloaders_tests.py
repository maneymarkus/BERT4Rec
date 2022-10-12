import abc
from absl import logging
import numpy as np
import tensorflow as tf

from bert4rec.dataloaders import get_dataloader_factory
from bert4rec.dataloaders import BERT4RecDataloader, BERT4RecML1MDataloader, BERT4RecML20MDataloader, \
    BERT4RecIMDBDataloader, BERT4RecRedditDataloader
import tests.test_utils as utils


class BaseBERT4RecDataloaderTests(abc.ABC):
    def __init__(self):
        self.dataloader = None

    def instantiate_dataloader(self, dataset: str):
        dataloader_factory = get_dataloader_factory("bert4rec")
        if dataset == "ml_1m":
            self.dataloader = dataloader_factory.create_ml_1m_dataloader()
        elif dataset == "ml_20m":
            self.dataloader = dataloader_factory.create_ml_20m_dataloader()
        elif dataset == "imdb":
            self.dataloader = dataloader_factory.create_imdb_dataloader()
        elif dataset == "reddit":
            self.dataloader = dataloader_factory.create_reddit_dataloader()
        else:
            raise ValueError(f"{dataset} is not a known dataset! And therefore the responsible dataloader "
                             f"can't be instantiated.")

    @abc.abstractmethod
    def test_preprocessing_regular(self):
        pass

    @abc.abstractmethod
    def test_preprocessing_without_masking(self):
        pass

    @abc.abstractmethod
    def test_dataloader_instance(self):
        pass

    @abc.abstractmethod
    def test_load_data(self):
        pass

    @abc.abstractmethod
    def test_generate_vocab(self):
        pass


class BERT4RecDataloaderTests(tf.test.TestCase):
    __test__ = False

    def setUp(self):
        super(BERT4RecDataloaderTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def test_dataloader_instance_general(self):
        if not hasattr(self, "dataloader"):
            return

        self.assertIsInstance(self.dataloader, BERT4RecDataloader,
                              f"The instantiated dataloader should be an instance of BERT4RecDataloader but is:"
                              f"{type(self.dataloader)}")

    def test_load_data_general(self):
        if not hasattr(self, "dataloader"):
            return

        ds = self.dataloader.load_data_into_ds()
        self.assertIsInstance(ds, tf.data.Dataset,
                              f"The loaded and returned dataset should be an instance of the tf.data.Dataset class "
                              f"but is an instance of: {type(ds)}")
        ds_item = ds.shuffle(10000, seed=3).take(1)
        # this statement retrieves the values of the dataset item converting tensors to numpy.ndarray objects
        ds_item_values = list(ds_item.as_numpy_iterator())[0]
        self.assertEqual(len(ds_item_values), 2,
                         f"A random item from the dataset should be a tuple of length 2 but is of length:"
                         f"{len(ds_item_values)}")
        self.assertIsInstance(ds_item_values[0], np.int64,
                              f"The first element from the dataset item tuple should be an np.int64 but is:"
                              f"{type(ds_item_values[0])}")
        self.assertIsInstance(ds_item_values[1], np.ndarray,
                              f"The second element from the dataset item tuple should be an np.ndarray but is:"
                              f"{type(ds_item_values[1])}")

    def test_preprocessing_regular_general(self, prepared_ds: tf.data.Dataset = None):
        if not hasattr(self, "dataloader") or prepared_ds is None:
            return

        self.assertIsInstance(prepared_ds, tf.data.Dataset,
                              f"The preprocessed dataset should actually be an instance of the tf.data.Dataset class"
                              f"but is an instance of: {type(prepared_ds)}")
        # no shuffle because filling the shuffle buffer with the preprocessed dataset takes ages
        ds_item = prepared_ds.take(1)
        # this statement retrieves the values of the dataset item converting tensors to numpy.ndarray objects
        ds_item_values = list(ds_item.as_numpy_iterator())[0]
        self.assertIsInstance(ds_item_values, dict,
                              f"An item of this preprocessed dataset from this particular dataloader should be"
                              f"a dict but is an instance of: {type(ds_item_values)}")
        return ds_item_values

    def test_feature_preprocessing_without_masking(self):
        general_dataloader = BERT4RecDataloader(10, 5)
        string_sequence_long = utils.generate_unique_word_list(size=15)
        processed_string_sequence_long = \
            general_dataloader.feature_preprocessing(None, string_sequence_long, apply_mlm=False)
        self.assertEqual(len(processed_string_sequence_long["input_word_ids"].numpy()), 10,
                         "The length of a longer input sequence should be truncated to _MAX_SEQ_LENGTH "
                         f"({general_dataloader._MAX_SEQ_LENGTH}) after calling `feature_processing()`, but "
                         f"the actual length is: {len(processed_string_sequence_long['input_word_ids'].numpy())}")
        string_sequence_short = utils.generate_unique_word_list(size=5)
        processed_string_sequence_short = \
            general_dataloader.feature_preprocessing(None, string_sequence_short, apply_mlm=False)
        self.assertEqual(len(processed_string_sequence_short["input_word_ids"].numpy()), 10,
                         "The length of a shorter input sequence should be extended (padded) to _MAX_SEQ_LENGTH "
                         f"({general_dataloader._MAX_SEQ_LENGTH}) after calling `feature_processing()`, but "
                         f"the actual length is: {len(processed_string_sequence_short['input_word_ids'].numpy())}")
        self.assertEqual(processed_string_sequence_short["input_word_ids"].numpy()[0], 0,
                         f"The first item of a originally shorter input sequence should be the pad token id "
                         f"({general_dataloader._PAD_TOKEN_ID}) but actually is: "
                         f"{processed_string_sequence_short['input_word_ids'].numpy()[0]}")
        self.assertEqual(processed_string_sequence_short["input_word_ids"].numpy()[4], 0,
                         f"The fifth item of a originally shorter input sequence should be the pad token id "
                         f"({general_dataloader._PAD_TOKEN_ID}) but actually is: "
                         f"{processed_string_sequence_short['input_word_ids'].numpy()[4]}")

    def test_prepare_inference(self):
        general_dataloader = BERT4RecDataloader(128, 5)
        # sequence is longer than maximum sequence length of ML model. If it works in this case it will work in
        # other cases too
        sequence = utils.generate_unique_word_list(size=140)
        preprocessed_sequence = general_dataloader.prepare_inference(sequence)
        # get the numpy value lists without any batches
        labels = preprocessed_sequence["labels"].numpy()[-1].tolist()
        input_word_ids = preprocessed_sequence["input_word_ids"].numpy()[-1].tolist()
        masked_lm_ids = preprocessed_sequence["masked_lm_ids"].numpy()[-1].tolist()
        masked_lm_weights = preprocessed_sequence["masked_lm_weights"].numpy()[-1].tolist()
        masked_lm_positions = preprocessed_sequence["masked_lm_positions"].numpy()[-1].tolist()
        print(preprocessed_sequence)
        self.assertListEqual(labels, input_word_ids,
                             f"The labels and input_word_ids tensors (or their values) should be equal, since the"
                             f"masking token (in the last place is a 'ground truth' label."
                             f"Label tensor values: {labels}. Input_word_ids tensor values: {input_word_ids}")
        self.assertListEqual(masked_lm_ids, [0],
                             f"The masked_lm_ids array should be equal to the {[0]} list, because the ground truth "
                             f"data for this particular token is a [PAD] token (0) as we don't know the actual "
                             f"value (since we're currently trying to infer it). But it actually is: {masked_lm_ids}")
        self.assertListEqual(masked_lm_weights, [1],
                             f"The masked_lm_weights array should be equal to the {[1]} list, but actually is: "
                             f"{masked_lm_weights}")
        self.assertEqual(len(masked_lm_positions), 1,
                         f"The length of the masked_lm_positions array should be 1 (as there is only one masked "
                         f"token at the end, but actually is: {len(masked_lm_positions)}")


class BERT4RecML1MDataloaderTests(BaseBERT4RecDataloaderTests, BERT4RecDataloaderTests):
    __test__ = True

    def setUp(self):
        super(BERT4RecML1MDataloaderTests, self).setUp()
        self.instantiate_dataloader("ml_1m")

    def tearDown(self):
        pass

    def test_dataloader_instance(self):
        super(BERT4RecML1MDataloaderTests, self).test_dataloader_instance_general()
        self.assertIsInstance(self.dataloader, BERT4RecML1MDataloader,
                              f"This specifically instantiated dataloader should be an instance of the"
                              f"BERT4RecML1MDataloader class but is:"
                              f"{type(self.dataloader)}")

    def test_load_data(self):
        super(BERT4RecML1MDataloaderTests, self).test_load_data_general()

    def test_generate_vocab(self):
        value = self.dataloader.generate_vocab()
        self.assertTrue(value, f"Calling the generate_vocab() method should simply return true")
        tokenizer = self.dataloader.get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, 3712, f"The vocab size of the tokenizer should be 3712 after calling the"
                                           f"generate_vocab() method but is: {vocab_size}.")

    def test_preprocessing_regular(self):
        prepared_ds = self.dataloader.preprocess_dataset()
        ds_item_values = super(BERT4RecML1MDataloaderTests, self).test_preprocessing_regular_general(prepared_ds)

        self.assertEqual(len(ds_item_values), 7,
                         f"An item of this preprocessed dataset (with default preprocessing) from this particular "
                         f"dataloader should be a dict with 7 entries but actually has: {len(ds_item_values)}")
        expected_keys = ['masked_lm_ids', 'masked_lm_positions', 'masked_lm_weights', 'labels',
                         'input_word_ids', 'input_mask', 'input_type_ids']
        self.assertListEqual(list(ds_item_values.keys()), expected_keys,
                             f"A dataset item (dict) should have all of these labels as keys: {expected_keys}")
        for key, value in ds_item_values.items():
            self.assertIsInstance(value, np.ndarray,
                                  f"One element of the dict of a single dataset element should actually be an"
                                  f"instance of numpy.ndarray but is: {type(value)}")
            self.assertIsInstance(value[0], np.ndarray,
                                  f"One element of the dict of a single dataset element should actually be at least"
                                  f"a 2-dimensional numpy.ndarray but the nested value is of type: {type(value[0])}")
        # could be extended like so: masked language model emits masked_lm_ids, masked_lm_positions
        # (and masked_lm_weights). All values of masked_lm_ids should be in labels and all values of
        # masked_lm_positions should be bigger than 0 and smaller than the size of the labels array (more
        # specifically: the positions should also only mask non-Pad and non-special tokens). The arrays
        # input_word_ids, input_type_ids and labels should start and end with special tokens (and may have pad tokens).
        # The array input_word_ids should contain mask token ids (randomly)

    def test_preprocessing_without_masking(self):
        # TODO
        pass


class BERT4RecML20MDataloaderTests(BaseBERT4RecDataloaderTests, BERT4RecDataloaderTests):
    __test__ = True

    def setUp(self):
        super(BERT4RecML20MDataloaderTests, self).setUp()
        self.instantiate_dataloader("ml_20m")

    def tearDown(self):
        pass

    def test_dataloader_instance(self):
        self.assertIsInstance(self.dataloader, BERT4RecDataloader,
                              f"The instantiated dataloader should be an instance of BERT4RecDataloader but is:"
                              f"{type(self.dataloader)}")
        self.assertIsInstance(self.dataloader, BERT4RecML20MDataloader,
                              f"This specifically instantiated dataloader should be an instance of the"
                              f"BERT4RecML20MDataloader class but is:"
                              f"{type(self.dataloader)}")

    def test_load_data(self):
        # requires a lot of time (on a Ryzen 5 5600x about 140 seconds)
        super(BERT4RecML20MDataloaderTests, self).test_load_data_general()

    def test_generate_vocab(self):
        value = self.dataloader.generate_vocab()
        self.assertTrue(value, f"Calling the generate_vocab() method should simply return true")
        tokenizer = self.dataloader.get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, 26735, f"The vocab size of the tokenizer should be 26735 after calling the"
                                            f"generate_vocab() method but is: {vocab_size}.")

    def test_preprocessing_regular(self):
        prepared_ds = self.dataloader.preprocess_dataset()
        ds_item_values = super(BERT4RecML20MDataloaderTests, self).test_preprocessing_regular_general(prepared_ds)

        self.assertEqual(len(ds_item_values), 7,
                         f"An item of this preprocessed dataset (with default preprocessing) from this particular "
                         f"dataloader should be a dict with 7 entries but actually has: {len(ds_item_values)}")
        expected_keys = ['masked_lm_ids', 'masked_lm_positions', 'masked_lm_weights', 'labels',
                         'input_word_ids', 'input_mask', 'input_type_ids']
        self.assertListEqual(list(ds_item_values.keys()), expected_keys,
                             f"A dataset item (dict) should have all of these labels as keys: {expected_keys}")
        for key, value in ds_item_values.items():
            self.assertIsInstance(value, np.ndarray,
                                  f"One element of the dict of a single dataset element should actually be an"
                                  f"instance of numpy.ndarray but is: {type(value)}")
            self.assertIsInstance(value[0], np.ndarray,
                                  f"One element of the dict of a single dataset element should actually be at least"
                                  f"a 2-dimensional numpy.ndarray but the nested value is of type: {type(value[0])}")

    def test_preprocessing_without_masking(self):
        # TODO
        pass


class BERT4RecIMDBDataloaderTests(BaseBERT4RecDataloaderTests, BERT4RecDataloaderTests):
    __test__ = True

    def setUp(self):
        super(BERT4RecIMDBDataloaderTests, self).setUp()
        self.instantiate_dataloader("imdb")

    def tearDown(self):
        pass

    def test_dataloader_instance(self):
        self.assertIsInstance(self.dataloader, BERT4RecDataloader,
                              f"The instantiated dataloader should be an instance of BERT4RecDataloader but is:"
                              f"{type(self.dataloader)}")
        self.assertIsInstance(self.dataloader, BERT4RecIMDBDataloader,
                              f"This specifically instantiated dataloader should be an instance of the"
                              f"BERT4RecIMDBDataloader class but is:"
                              f"{type(self.dataloader)}")

    def test_load_data(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.load_data_into_ds()

    def test_generate_vocab(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.generate_vocab()

    def test_preprocessing_regular(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.preprocess_dataset()

    def test_preprocessing_without_masking(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.preprocess_dataset(apply_mlm=False)


class BERT4RecRedditDataloaderTests(BaseBERT4RecDataloaderTests, BERT4RecDataloaderTests):
    __test__ = True

    def setUp(self):
        super(BERT4RecRedditDataloaderTests, self).setUp()
        self.instantiate_dataloader("reddit")

    def tearDown(self):
        pass

    def test_dataloader_instance(self):
        self.assertIsInstance(self.dataloader, BERT4RecDataloader,
                              f"The instantiated dataloader should be an instance of BERT4RecDataloader but is:"
                              f"{type(self.dataloader)}")
        self.assertIsInstance(self.dataloader, BERT4RecRedditDataloader,
                              f"This specifically instantiated dataloader should be an instance of the"
                              f"BERT4RecRedditDataloader class but is:"
                              f"{type(self.dataloader)}")

    def test_load_data(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.load_data_into_ds()

    def test_generate_vocab(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.generate_vocab()

    def test_preprocessing_regular(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.preprocess_dataset()

    def test_preprocessing_without_masking(self):
        # This method of this dataloader is expected to raise an error still
        with self.assertRaises(NotImplementedError):
            self.dataloader.preprocess_dataset(apply_mlm=False)


if __name__ == "__main__":
    tf.test.main()
