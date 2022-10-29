from absl import logging
import pathlib
import random
import tensorflow as tf

from bert4rec.dataloaders import BERT4RecDataloader
import bert4rec.dataloaders.dataloader_utils as dataloader_utils
from bert4rec.model import BERTModel
from bert4rec.model.components import networks
import bert4rec.model.model_utils as utils
import tests.test_utils as test_utils


class ModelUtilsTests(tf.test.TestCase):
    def setUp(self):
        super(ModelUtilsTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        pass

    def _build_model(self, vocab_size: int):
        bert_encoder = networks.BertEncoder(vocab_size=vocab_size)
        return BERTModel(bert_encoder)

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

        absolute_path = path.absolute()
        determined_absolute_path = utils.determine_model_path(absolute_path, 1)
        self.assertEqual(absolute_path, determined_absolute_path,
                         f"Determining the model path if an absolute path is given should always return "
                         f"the given absolute path without any modifications. "
                         f"Expected path: {absolute_path}. Actual path: {determined_absolute_path}")

    def test_rank_items(self):
        ds_size = 1

        prepared_ds, dataloader = test_utils.generate_random_sequence_dataset(ds_size)
        prepared_batches = dataloader_utils.make_batches(prepared_ds)
        vocab_size = dataloader.tokenizer.get_vocab_size()

        model = self._build_model(vocab_size)
        # initialize weights
        _ = model(model.inputs)

        encoder_input = None
        for el in prepared_batches.take(1):
            encoder_input = el
        random_rank_items = [random.randint(0, vocab_size) for _ in range(5)]
        gathered_item_embeddings = tf.gather(model.encoder.get_embedding_table(), random_rank_items)

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

            correct_ranking_indexes = tf.argsort(vocab_probabilities, direction="DESCENDING")
            correct_ranking = tf.gather(random_rank_items, correct_ranking_indexes)
            highest_ranked_item = correct_ranking[0].numpy()
            lowest_ranked_item = correct_ranking[len(correct_ranking) - 1].numpy()

            self.assertEqual(ranking[0].numpy(), highest_ranked_item,
                             f"The item with the highest probability ({highest_ranked_item}) should actually be "
                             f"the first item in the ranking list, but the first item is: {ranking[0]}")

            self.assertEqual(ranking[len(ranking) - 1].numpy(), lowest_ranked_item,
                             f"The item with the lowest probability ({lowest_ranked_item}) should actually be "
                             f"the last item in the ranking list, but the last item is: "
                             f"{ranking[len(ranking) - 1]}")

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
