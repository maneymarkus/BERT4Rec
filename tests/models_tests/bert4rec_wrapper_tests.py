from absl import logging
import pathlib
import random
import tempfile
import tensorflow as tf

from bert4rec.dataloaders import BERT4RecDataloader, dataloader_utils
from bert4rec.models import BERT4RecModel, BERT4RecModelWrapper
from bert4rec.models.bert4rec_wrapper import _META_CONFIG_FILE_NAME, _TOKENIZER_VOCAB_FILE_NAME
from bert4rec.models.components import networks
from bert4rec import tokenizers
from bert4rec.trainers import optimizers
import tests.test_utils as test_utils


class BERT4RecWrapperTests(tf.test.TestCase):
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
                     optimizer: tf.keras.optimizers.Optimizer = None,
                     max_sequence_len: int = 10):
        # instantiate just a very small encoder for testing purposes
        bert_encoder = networks.Bert4RecEncoder(vocab_size=vocab_size,
                                                hidden_size=16,
                                                num_layers=2,
                                                max_sequence_length=max_sequence_len,
                                                inner_dim=16,
                                                num_attention_heads=2)
        bert_model = BERT4RecModel(bert_encoder)
        if optimizer is None:
            optimizer = self.optimizer
        bert_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        # makes sure the weights are built
        _ = bert_model(bert_model.inputs)
        return bert_model

    def test_save_model(self):
        ds_size = 1
        vocab_size = 200
        max_sequence_length = 10
        tmpdir = tempfile.TemporaryDirectory()
        save_path = pathlib.Path(tmpdir.name)

        bert_model = self._build_model(vocab_size, max_sequence_len=max_sequence_length)
        wrapper = BERT4RecModelWrapper(bert_model)

        # makes sure the weights are built
        _ = bert_model(bert_model.inputs)

        # should throw an error when trying to save a not fully initialized models
        # i.e. models can't be saved without at least one training step as this builds the loss and metrics
        with self.assertRaises(RuntimeError):
            wrapper.save(save_path)

        dataloader = BERT4RecDataloader(max_seq_len=max_sequence_length, max_predictions_per_seq=5)
        dataset, _ = test_utils.generate_random_sequence_dataset(ds_size, dataloader=dataloader)
        batches = dataloader_utils.make_batches(dataset)
        random_input = None
        for el in batches.take(1):
            random_input = el

        # the one train step to build the compiled metrics and loss
        bert_model.train_step(random_input)

        # save only models and meta config
        wrapper.save(save_path)
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

        tokenizer = tokenizers.get()
        # makes sure, vocab has at least one entry
        tokenizer.tokenize("test")

        # save models and tokenizer vocab and meta config
        wrapper.save(save_path, tokenizer)
        vocab_path = save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME)

        self.assertTrue(vocab_path.is_file(),
                        f"the saved model directory should contain a '{_TOKENIZER_VOCAB_FILE_NAME}' file.")

    def test_load_model(self):
        ds_size = 1
        vocab_size = 200
        max_sequence_length = 10
        tmpdir = tempfile.TemporaryDirectory()
        save_path = pathlib.Path(tmpdir.name)

        bert_model = self._build_model(vocab_size, max_sequence_len=max_sequence_length)
        wrapper = BERT4RecModelWrapper(bert_model)

        dataloader = BERT4RecDataloader(max_seq_len=max_sequence_length, max_predictions_per_seq=5)
        dataset, _ = test_utils.generate_random_sequence_dataset(ds_size, dataloader=dataloader)
        batches = dataloader_utils.make_batches(dataset)
        random_input = None
        for el in batches.take(1):
            random_input = el

        # also test reloading of tokenizer
        tokenizer = tokenizers.get()
        # makes sure that the tokenizer has a vocab of at least one word
        tokenizer.tokenize("word")

        # the one train step to build the compiled metrics and loss
        bert_model.train_step(random_input)

        wrapper.save(save_path, tokenizer)

        del wrapper, bert_model, tokenizer

        loaded_assets = BERT4RecModelWrapper.load(save_path)
        reloaded_wrapper = loaded_assets["model_wrapper"]
        reloaded_bert_model = reloaded_wrapper.bert_model
        reloaded_encoder = reloaded_bert_model.encoder
        tokenizer = loaded_assets["tokenizer"]

        self.assertIsInstance(reloaded_wrapper, BERT4RecModelWrapper,
                              f"The loaded model wrapper should be an instance of {BERT4RecModelWrapper}, "
                              f"but is an instance of: {type(reloaded_wrapper)}")
        # assertions fail because (re)loading a saved models does not instantiate the original class yet
        self.assertIsInstance(reloaded_bert_model, BERT4RecModel,
                              f"The loaded (in the model wrapper contained) model should be an instance of "
                              f"{BERT4RecModel} but is actually an instance of: {type(reloaded_bert_model)}")
        self.assertIsInstance(reloaded_encoder, networks.Bert4RecEncoder,
                              f"The loaded encoder layer (contained in the loaded model) should be an instance of "
                              f"{networks.Bert4RecEncoder} but actually is an instance of: "
                              f"{type(reloaded_encoder)}")
        self.assertIsInstance(tokenizer, tokenizers.SimpleTokenizer,
                              f"Saving a model with a tokenizer should actually also save the tokenizer "
                              f"vocab and reload it as well again. In this case a new instance of "
                              f"{tokenizers.SimpleTokenizer} should have been created with the vocab restored, "
                              f"but actually got: {tokenizer}")
        self.assertEqual(tokenizer.get_vocab_size(), 1,
                         f"Reloading the tokenizer along with the saved model should restore its vocab. "
                         f"Expected vocab size: 1. Restored vocab size: {tokenizer.get_vocab_size()}")

    def test_rank_items(self):
        ds_size = 1
        number_rank_items = 7
        max_sequence_length = 50

        dataloader = BERT4RecDataloader(max_seq_len=max_sequence_length, max_predictions_per_seq=5)
        dataset, _ = test_utils.generate_random_sequence_dataset(ds_size, dataloader=dataloader)
        batches = dataloader_utils.make_batches(dataset)
        vocab_size = dataloader.tokenizer.get_vocab_size()

        bert_model = self._build_model(vocab_size, max_sequence_len=max_sequence_length)
        wrapper = BERT4RecModelWrapper(bert_model)

        # initialize weights
        _ = bert_model(bert_model.inputs)

        encoder_input = None
        for el in batches.take(1):
            encoder_input = el

        # use just one generic item list for ranking for all masked tokens
        rank_items_1 = [random.randint(0, vocab_size) for _ in range(number_rank_items)]

        rankings_1, probabilities_1 = wrapper.rank(encoder_input,
                                                   rank_items_1,
                                                   encoder_input["masked_lm_positions"])

        self.assertEqual(len(rankings_1[0]), len(probabilities_1[0]),
                         f"The length of the ranking list ({len(rankings_1[0])}) should be equal to the "
                         f"probabilities list length ({len(probabilities_1[0])})")
        self.assertEqual(len(rankings_1[0]), len(encoder_input["masked_lm_positions"][0]),
                         f"The amount of individual rankings should be equal to the amount of masked_lm_positions "
                         f"({len(encoder_input['masked_lm_positions'][0])}) but received {len(rankings_1)} "
                         f"individual rankings.")
        ranking = random.choice(rankings_1[0])
        probability = random.choice(probabilities_1[0])
        self.assertEqual(len(ranking), len(probability),
                         f"An individual ranking should have the same amount of items as the respective probability "
                         f"distribution ({len(probability)}) but actually has: {len(ranking)}")
        self.assertEqual(len(ranking), number_rank_items,
                         f"An individual ranking should have as many items as the original rank_items list "
                         f"({len(rank_items_1)}) but actually has: {len(ranking)}")
        self.assertAllInSet(ranking, set(rank_items_1))

        # use individual ranking lists for each token -> shape of rank_items list: (batch, tokens, rank_items)
        rank_items_2 = [
            [
                [random.randint(0, vocab_size) for _ in range(random.randint(3, 10))]
                for _ in range(len(encoder_input["masked_lm_positions"][0]))
            ]
        ]

        rankings_2, probabilities_2 = wrapper.rank(encoder_input,
                                                   rank_items_2,
                                                   encoder_input["masked_lm_positions"])

        # iterate over batches
        for b in range(len(rankings_2)):
            # iterate over tokens
            for i, token_idx in enumerate(rankings_2[b]):
                self.assertEqual(len(rankings_2[b][i]), len(probabilities_2[b][i]),
                                 f"The length of an individual ranking list ({len(rankings_2[b][i])}) should be "
                                 f"equal to the probabilities list length ({len(probabilities_2[b][i])})")
                self.assertEqual(len(rankings_2[b][i]), len(rank_items_2[b][i]),
                                 f"The length of an individual ranking list ({len(rankings_2[b][i])}) should be "
                                 f"equal to the corresponding original (unranked) items list ("
                                 f"{len(rank_items_2[b][i])})")

    def test_update_meta(self):
        vocab_size = 200

        new_entries = {
            "new_entry": "test",
            "some": "info",
        }

        bert_model = self._build_model(vocab_size)
        wrapper = BERT4RecModelWrapper(bert_model)

        wrapper.update_meta(new_entries)
        meta_config = wrapper.get_meta_config()
        self.assertIn("new_entry", meta_config,
                      f"'new_entry' should be a key present in the meta config of the wrapper class "
                      f"but it has only these keys: {wrapper.get_meta_config().keys()}.")
        self.assertIn("some", meta_config,
                      f"'some' should be a key present in the meta config of the wrapper class "
                      f"but it has only these keys: {wrapper.get_meta_config().keys()}.")
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
        vocab_size = 200

        bert_model = self._build_model(vocab_size)
        wrapper = BERT4RecModelWrapper(bert_model)

        meta_config = wrapper.get_meta_config()
        meta_keys = list(meta_config.keys())
        random_meta_key = random.choice(meta_keys)
        logging.debug(random_meta_key)
        wrapper.delete_keys_from_meta(random_meta_key)
        self.assertNotIn(random_meta_key, wrapper.get_meta_config(),
                         f"Deleting a key ({random_meta_key}) from the meta config "
                         f"should remove it from the meta config dict, but it is still available in: "
                         f"{wrapper.get_meta_config()}")

        if len(meta_config) < 3:
            wrapper.update_meta({"key1": "value1", "key2": "value2", "key3": "value3"})

        meta_keys = list(meta_config.keys())
        random_meta_keys = [random.choice(meta_keys) for _ in range(2)]
        wrapper.delete_keys_from_meta(random_meta_keys)
        for rmk in random_meta_keys:
            self.assertNotIn(rmk, wrapper.get_meta_config(),
                             f"Deleting a list of keys ({random_meta_keys}) should remove both keys from "
                             f"the meta config dict, but the key '{rmk}' is still available in: "
                             f"{wrapper.get_meta_config()}")


if __name__ == "__main__":
    tf.test.main()
