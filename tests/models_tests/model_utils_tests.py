from absl import logging
import pathlib
import random
import tensorflow as tf

from bert4rec.dataloaders import BERT4RecDataloader
from bert4rec.model import BERT4RecModel
import bert4rec.model.model_utils as utils
import tests.test_utils as test_utils


class ModelUtilsTests(tf.test.TestCase):
    def setUp(self):
        super(ModelUtilsTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        pass

    def _instantiate_dataloader(self):
        return BERT4RecDataloader()

    def _build_model(self, vocab_size: int):
        return BERT4RecModel(vocab_size=vocab_size)

    def test_determine_model_path(self):
        path = pathlib.Path(".")
        with self.assertRaises(ValueError):
            utils.determine_model_path(path, -1)
            utils.determine_model_path(path, 3)
        determined_path_1 = utils.determine_model_path(path, 2)
        self.assertEqual(determined_path_1, path,
                         f"Calling the method with mode 2 should return the same path (and thus be relative "
                         f"to the current working directory. "
                         f"Expected path: {path}, "
                         f"determined path: {determined_path_1}")
        dir_name = "saved_models"
        path = pathlib.Path("saved_models")
        determined_path_2 = utils.determine_model_path(path, 0)
        self.assertTrue(determined_path_2.is_absolute(),
                        f"The method should return an absolute path in mode 0 (with the original path added "
                        f"on top of the project directory path)")
        determined_path_3 = utils.determine_model_path(path, 1)
        self.assertTrue(determined_path_3.is_absolute(),
                        f"The method should return an absolute path in mode 0 (with the original path added "
                        f"on top of the virtual environment path)")
        self.assertEqual(determined_path_2.stem, dir_name,
                         f"Determining the absolute path should always result in only a longer path with the "
                         f"original path added to it. "
                         f"Expected last path element: {dir_name}, "
                         f"Actual last path element: {determined_path_2.stem}")

    def test_rank_items(self):
        vocab_size = 500
        model = self._build_model(vocab_size)
        dataloader = self._instantiate_dataloader()

        random_sequence = test_utils.generate_unique_word_list(size=50)
        random_rank_items = [random.randint(0, vocab_size) for _ in range(5)]
        gathered_item_embeddings = tf.gather(model.encoder.get_embedding_table(), random_rank_items)

        encoder_input = dataloader.feature_preprocessing(None, random_sequence)
        encoder_output = model(encoder_input)
        sequence_output = encoder_output["sequence_output"]
        pooled_output = encoder_output["pooled_output"][0]

        probabilities = list()
        rankings = list()
        for token_index in encoder_input["masked_lm_positions"][0]:
            token_logits = sequence_output[0, token_index, :]
            vocab_probabilities, ranking = utils.rank_items(token_logits,
                                                            gathered_item_embeddings,
                                                            random_rank_items)
            probabilities.append(vocab_probabilities)
            rankings.append(ranking)

        self.assertEqual(len(probabilities), len(encoder_input["masked_lm_positions"][0]),
                         f"The length of the generated probabilities list should have the same length as "
                         f"the masked_lm_positions encoder input. "
                         f"Expected length: {len(encoder_input['masked_lm_positions'][0])} "
                         f"Actual length: {len(probabilities)}")
        self.assertEqual(len(rankings), len(encoder_input["masked_lm_positions"][0]),
                         f"The length of the generated rankings list should have the same length as "
                         f"the masked_lm_positions encoder input. "
                         f"Expected length: {len(encoder_input['masked_lm_positions'][0])} "
                         f"Actual length: {len(rankings)}")

        for r in rankings:
            # Each ranking should have the same elements as the original unranked list just in another order.
            self.assertAllInSet(r, set(random_rank_items))

        with self.assertRaises(AssertionError):
            # test some triggers of the AssertionError
            utils.rank_items(pooled_output[:-1], gathered_item_embeddings, random_rank_items)
            utils.rank_items(pooled_output, gathered_item_embeddings[:, :-1], random_rank_items)
            utils.rank_items(pooled_output, gathered_item_embeddings[:-1], random_rank_items)
            utils.rank_items(pooled_output, gathered_item_embeddings, random_rank_items[:-1])


if __name__ == "__main__":
    tf.test.main()