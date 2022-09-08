from absl import logging
import pathlib
import random
import string
import tempfile
import tensorflow as tf

from bert4rec.dataloaders import BERT4RecDataloader
from bert4rec.model.bert4rec_model import BERTModel, BERT4RecModelWrapper, \
    _META_CONFIG_FILE_NAME, _TOKENIZER_VOCAB_FILE_NAME
from bert4rec.model.components import networks
from bert4rec.tokenizers import SimpleTokenizer
from bert4rec.trainers import optimizers
import tests.test_utils as utils


class BERT4RecModelTests(tf.test.TestCase):
    # Warning: May take some time
    def setUp(self):
        super(BERT4RecModelTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)
        self.vocab_size = 300
        optimizer_factory = optimizers.get_optimizer_factory("bert4rec")
        self.optimizer = optimizer_factory.create_adam_w_optimizer()

        self.encoder = networks.BertEncoder(self.vocab_size)
        self.bert_model = BERTModel(self.encoder)
        self.bert_model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        self.bert4rec_wrapper = BERT4RecModelWrapper(self.bert_model)

    def tearDown(self):
        self.vocab_size = None
        self.encoder = None
        self.bert_model = None
        self.bert4rec_wrapper = None

    def test_load_model(self):
        tmpdir = tempfile.TemporaryDirectory()
        save_path = pathlib.Path(tmpdir.name)

        _ = self.bert_model(self.bert_model.inputs)

        # should throw an error when trying to save a not fully initialized model
        with self.assertRaises(RuntimeError):
            self.bert4rec_wrapper.save(save_path)

        # initialize model fully by building the loss
        random_sequence = [random.choice(string.ascii_letters) for _ in range(100)]
        dataloader = BERT4RecDataloader()
        inputs = dataloader.feature_preprocessing(None, random_sequence)
        self.bert_model.train_step(inputs)

        self.bert4rec_wrapper.save(save_path)

        self.bert4rec_wrapper = None
        self.bert_model = None
        self.encoder = None

        # reloading just the model
        loaded_assets = BERT4RecModelWrapper.load(save_path)
        self.bert4rec_wrapper = loaded_assets["model"]
        self.bert_model = self.bert4rec_wrapper.bert_model
        self.encoder = self.bert_model.encoder

        self.assertIsInstance(self.bert4rec_wrapper, BERT4RecModelWrapper,
                              f"The loaded model wrapper should be an instance of {BERT4RecModelWrapper}, "
                              f"but is an instance of: {type(self.bert4rec_wrapper)}")
        # assertions fail because (re)loading a saved model does not instantiate the original class yet
        self.assertIsInstance(self.bert_model, BERTModel,
                              f"The loaded (in the model wrapper contained) model should be an instance of "
                              f"{BERTModel} but is actually an instance of: {type(self.bert_model)}")
        self.assertIsInstance(self.encoder, networks.BertEncoder,
                              f"The loaded encoder layer (contained in the loaded model) should be an instance of "
                              f"{networks.BertEncoder} but actually is an instance of: "
                              f"{type(self.encoder)}")

        # reloading a model with tokenizer

        self.encoder = networks.BertEncoder(self.vocab_size)
        self.bert_model = BERTModel(self.encoder)
        self.bert4rec_wrapper = BERT4RecModelWrapper(self.bert_model)
        tokenizer = SimpleTokenizer()
        # makes sure that the tokenizer has a vocab of at least one word
        tokenizer.tokenize("word")

        # makes sure the weights are built (note: the model is not compiled yet; weights are necessary for saving
        # compilation info not)
        _ = self.bert_model(self.bert_model.inputs)

        self.bert4rec_wrapper.save(save_path, tokenizer)

        self.bert4rec_wrapper = None
        self.bert_model = None
        self.encoder = None
        tokenizer = None

        loaded_assets = BERT4RecModelWrapper.load(save_path)
        self.bert4rec_wrapper = loaded_assets["model"]
        self.bert_model = self.bert4rec_wrapper.bert_model
        self.encoder = self.bert_model.encoder
        tokenizer = loaded_assets["tokenizer"]

        self.assertIsInstance(self.bert4rec_wrapper, BERT4RecModelWrapper,
                              f"The loaded model wrapper should be an instance of {BERT4RecModelWrapper}, "
                              f"but is an instance of: {type(self.bert4rec_wrapper)}")
        # assertions fail because (re)loading a saved model does not instantiate the original class yet
        self.assertIsInstance(self.bert_model, BERTModel,
                              f"The loaded (in the model wrapper contained) model should be an instance of "
                              f"{BERTModel} but is actually an instance of: {type(self.bert_model)}")
        self.assertIsInstance(self.encoder, networks.BertEncoder,
                              f"The loaded encoder layer (contained in the loaded model) should be an instance of "
                              f"{networks.BertEncoder} but actually is an instance of: "
                              f"{type(self.encoder)}")
        self.assertIsInstance(tokenizer, SimpleTokenizer,
                              f"Saving a model with a tokenizer should actually also save the tokenizer "
                              f"vocab and reload it as well again. In this case a new instance of "
                              f"{SimpleTokenizer} should have been created with the vocab restored.")
        self.assertEqual(tokenizer.get_vocab_size(), 1,
                         f"Reloading the tokenizer along with the saved model should restore its vocab. "
                         f"Expected vocab size: 1. Restored vocab size: {tokenizer.get_vocab_size()}")

    def test_save_model(self):
        tmpdir = tempfile.TemporaryDirectory()
        save_path = pathlib.Path(tmpdir.name)
        # makes sure the weights are built
        _ = self.bert_model(self.bert_model.inputs)

        # save only model and meta config
        self.bert4rec_wrapper.save(save_path)
        assets_path = save_path.joinpath("assets")
        keras_metadata_path = save_path.joinpath("keras_metadata.pb")
        meta_config_path = save_path.joinpath(_META_CONFIG_FILE_NAME)
        saved_model_path = save_path.joinpath("saved_model.pb")
        variables_path = save_path.joinpath("variables")
        self.assertTrue(assets_path.is_dir(),
                        f"the saved model directory should contain an 'assets' directory.")
        self.assertTrue(keras_metadata_path.is_file(),
                        f"the saved model directory should contain a 'keras_metadata.pb' file.")
        self.assertTrue(meta_config_path.is_file(),
                        f"the saved model directory should contain a '{_META_CONFIG_FILE_NAME}' file.")
        self.assertTrue(saved_model_path,
                        f"the saved model directory should contain a 'saved_model.pb' file.")
        self.assertTrue(variables_path.is_dir(),
                        f"the saved model directory should contain a 'variables' directory.")

        tokenizer = SimpleTokenizer()
        # makes sure, vocab has at least one entry
        tokenizer.tokenize("test")

        # save model and tokenizer vocab and meta config
        self.bert4rec_wrapper.save(save_path, tokenizer)
        vocab_path = save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME)

        self.assertTrue(vocab_path.is_file(),
                        f"the saved model directory should contain a '{_TOKENIZER_VOCAB_FILE_NAME}' file.")

    def test_rank_items(self):
        number_rank_items = 7
        sequence_length = 50

        dataloader = BERT4RecDataloader()
        sequence = utils.generate_unique_word_list(size=sequence_length)

        encoder_input = dataloader.feature_preprocessing(None, sequence)

        rank_items = [random.randint(0, self.vocab_size) for _ in range(number_rank_items)]

        rankings, probabilities = self.bert4rec_wrapper.rank(encoder_input,
                                                             rank_items,
                                                             encoder_input["masked_lm_positions"])
        self.assertEqual(len(rankings), len(probabilities),
                         f"The length of the ranking list ({len(rankings)}) should be equal to the "
                         f"probabilities list length ({len(probabilities)})")
        self.assertEqual(len(rankings), len(encoder_input["masked_lm_positions"][0]),
                         f"The amount of individual rankings should be equal to the amount of masked_lm_positions "
                         f"({len(encoder_input['masked_lm_positions'][0])}) but received {len(rankings)} "
                         f"individual rankings.")
        ranking = random.choice(rankings)
        probability = random.choice(probabilities)
        self.assertEqual(len(ranking), len(probability),
                         f"An individual ranking should have the same amount of items as the respective probability "
                         f"distribution ({len(probability)}) but actually has: {len(ranking)}")
        self.assertEqual(len(ranking), number_rank_items,
                         f"An individual ranking should have as many items as the original rank_items list "
                         f"({len(rank_items)}) but actually has: {len(ranking)}")
        self.assertAllInSet(ranking, set(rank_items))

    def test_update_meta(self):
        new_entries = {
            "new_entry": "test",
            "some": "info",
        }
        self.bert4rec_wrapper.update_meta(new_entries)
        meta_config = self.bert4rec_wrapper.get_meta()
        self.assertIn("new_entry", meta_config,
                      f"'new_entry' should be a key present in the meta config of the wrapper class "
                      f"but it has only these keys: {self.bert4rec_wrapper.get_meta().keys()}.")
        self.assertIn("some", meta_config,
                      f"'some' should be a key present in the meta config of the wrapper class "
                      f"but it has only these keys: {self.bert4rec_wrapper.get_meta().keys()}.")
        self.assertEqual(new_entries["new_entry"], meta_config["new_entry"],
                         f"The values for the 'new_entry' key should be equal in the new_entries dict and "
                         f"the meta_config dict."
                         f"new_entries value: {new_entries['new_entry']}"
                         f"meta_config value: {meta_config['new_entry']}")
        self.assertEqual(new_entries["some"], meta_config["some"],
                         f"The values for the 'some' key should be equal in the new_entries dict and "
                         f"the meta_config dict."
                         f"new_entries value: {new_entries['some']}"
                         f"meta_config value: {meta_config['some']}")

    def test_delete_key_from_meta(self):
        meta_config = self.bert4rec_wrapper.get_meta()
        meta_keys = list(meta_config.keys())
        random_meta_key = random.choice(meta_keys)
        print(random_meta_key)
        self.bert4rec_wrapper.delete_keys_from_meta(random_meta_key)
        self.assertNotIn(random_meta_key, self.bert4rec_wrapper.get_meta(),
                         f"Deleting a key ({random_meta_key}) from the meta config "
                         f"should remove it from the meta config dict, but it is still available in: "
                         f"{self.bert4rec_wrapper.get_meta()}")

        if len(meta_config) < 3:
            self.bert4rec_wrapper.update_meta({"key1": "value1", "key2": "value2", "key3": "value3"})

        meta_keys = list(meta_config.keys())
        random_meta_keys = [random.choice(meta_keys) for _ in range(2)]
        self.bert4rec_wrapper.delete_keys_from_meta(random_meta_keys)
        for rmk in random_meta_keys:
            self.assertNotIn(rmk, self.bert4rec_wrapper.get_meta(),
                             f"Deleting a list of keys ({random_meta_keys}) should remove both keys from "
                             f"the meta config dict, but the key '{rmk}' is still available in: "
                             f"{self.bert4rec_wrapper.get_meta()}")


if __name__ == "__main__":
    tf.test.main()
