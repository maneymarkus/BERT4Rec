from absl import logging
import json
import pathlib
import random
import tensorflow as tf
from typing import Union

from bert4rec.dataloaders.bert4rec_dataloaders import BERT4RecML1MDataloader
from bert4rec.model.components import networks
import bert4rec.model.model_utils as utils
from bert4rec.tokenizers import BaseTokenizer, tokenizer_factory


_META_CONFIG_FILE_NAME = "meta_config.json"
_TOKENIZER_VOCAB_FILE_NAME = "vocab.txt"


class BERT4RecModel(tf.keras.Model):
    def __init__(self,
                 encoder: networks.BertEncoder = None,
                 vocab_size: int = None,
                 **kwargs):
        if encoder is None:
            if vocab_size is None:
                raise ValueError("The vocab size has to be given, if no encoder is given!")
            encoder = networks.BertEncoder(vocab_size=vocab_size)

        inputs = encoder.inputs
        predictions = encoder(inputs)

        # parent call after (!) creation of the network with the Functional API
        super(BERT4RecModel, self).__init__(inputs=inputs, outputs=predictions, **kwargs)
        self.encoder = encoder
        # called it meta config, since tensorflow layers and models already have a proper config dict
        self._meta_config = dict(
            {
                "model": "BERT4Rec",
                "tokenizer": None,
                "last_trained": None,
                "trained_on_dataset": None
            }
        )

    @classmethod
    def load(cls, save_path: pathlib.Path, mode: int = 0):
        """
        Loads and returns all available assets in conjunction with the ml model. Loads at least a saved
        BERT4Rec model. Depending on the configuration might also load tokenizer with learnt vocab

        :param save_path: Path to the model directory
        :param mode: The mode determines from where the model is loaded. Available modes are 0, 1, 2.
        0 is default and loads the model relative from the project root. 1 loads the model relative
        from the virtual environment and 2 loads it relative from the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: Dict with all loaded assets
        """
        save_path = utils.determine_model_path(save_path, mode)
        loaded_assets = dict()
        loaded_model = tf.keras.models.load_model(save_path)
        loaded_assets["model"] = loaded_model

        try:
            with open(save_path.joinpath(_META_CONFIG_FILE_NAME)) as jf:
                meta_config = json.load(jf)
                loaded_model._meta_config = meta_config

                if "tokenizer" in meta_config and meta_config["tokenizer"] is not None:
                    tokenizer = tokenizer_factory.get_tokenizer(meta_config["tokenizer"])
                    tokenizer.import_vocab_from_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
                    loaded_assets["tokenizer"] = tokenizer

        except FileNotFoundError:
            logging.error(f"The meta configuration/information json file (meta_config.json) could not be found "
                          f"in the supposed model directory: {save_path}")

        return loaded_assets

    def save(self, save_path: pathlib.Path, tokenizer: BaseTokenizer = None, mode: int = 0) -> True:
        """
        Saves a model to the disk. If the tokenizer is given, saves the used vocab from the tokenizer as well.

        :param save_path: The pathto save the model to
        :param tokenizer: If given, saves the used vocab from this tokenizer in the model directory
        :param mode: The mode determines where the model is stored. Available modes are 0, 1, 2.
        0 is default and stores the model relative to the project root. 1 stores the model relative
        to the virtual environment and 2 stores it relative to the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: True
        """
        save_path = utils.determine_model_path(save_path, mode)
        super(BERT4RecModel, self).save(save_path)

        if tokenizer:
            tokenizer.export_vocab_to_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
            self.update_meta({"tokenizer": tokenizer.code})

        with open(save_path.joinpath(_META_CONFIG_FILE_NAME), "w") as f:
            json.dump(self._meta_config, f, indent=4)

        return True

    def rank(self, encoder_input: Union[list, tf.Tensor], rank_items: list, sequence_indexes: tf.Tensor = None):
        """
        Ranks a given set of potential items (rank_items) by probability based on the given sequence.
        The items to be ranked should actually be the indexes of the items to gather their embeddings
        from the embeddings table, so basically just their numeric tokens.

        :param encoder_input: The input to the encoder on which the ranking of the rank_items is based on.
        Should be the regular BERT input dictionary
        :param rank_items: Either list of indexes of the items which should be ranked (in case
        sequence_indexes param is not given) or a list of lists with the indexes of the ranking items
        (if sequence_indexes param is given). If this is a list of lists, the length of the primary list
        should be equal to the length of the sequence_indexes as each index in either list corresponds to
        the index of the other list
        :param sequence_indexes: A list of integers determining the output positions
        which should be used to (individually) rank the items (based on the corresponding output
        of the sequence_output from the encoder). If none os given, the items are ranked based on
        the accumulated encoder output (pooled_output). Could legitimately be the masked language model
        (or simply the positions of the masked tokens).
        :return: A tuple with the first value containing the ranked items from highest probability to lowest
        and the second value containing the probabilities (not sorted!)
        """
        gathered_embeddings = tf.gather(self.encoder.get_embedding_table(), rank_items)

        encoder_output = self(encoder_input)

        probabilities = list()
        rankings = list()
        if sequence_indexes is not None:
            sequence_output = encoder_output["sequence_output"]
            # iterate over the given sequence_indexes: should be in the shape of [batch, num_tokens] with batch = 1
            # since it's a specific task
            batch = 0
            for token_index in sequence_indexes[batch]:
                token_logits = sequence_output[batch, token_index, :]
                vocab_probabilities, ranking = utils.rank_items(token_logits, gathered_embeddings, rank_items)
                probabilities.append(vocab_probabilities)
                rankings.append(ranking)
        else:
            output_logits = encoder_output["pooled_output"][0, :]
            vocab_probabilities, ranking = utils.rank_items(output_logits, gathered_embeddings, rank_items)
            probabilities.append(vocab_probabilities)
            rankings.append(ranking)

        return rankings, probabilities

    def update_meta(self, updated_info: dict):
        self._meta_config.update(updated_info)


if __name__ == "__main__":
    dataloader = BERT4RecML1MDataloader()
    #dataloader.generate_vocab()
    ds = dataloader.preprocess_dataset()
    for element in ds.take(1):
        example = element
    #tokenizer = dataloader.get_tokenizer()

    model = BERT4RecModel(vocab_size=30522)
    model.compile(optimizer="adam", loss="mse")
    #embedding_table = model.encoder.get_embedding_table()
    #gathered_embeddings = tf.gather(embedding_table, [0, 3, 30519, 30521])
    #print(gathered_embeddings)
    #print(model.get_config())
    #model.save(pathlib.Path("my_model"), tokenizer)
    #loaded_assets = BERT4RecModel.load(pathlib.Path("my_model"))
    #loaded_tokenizer = loaded_assets["tokenizer"]
    #model = loaded_assets["model"]
    #print(model.encoder.get_embedding_table())

    #predictions = model(example)
    #print(predictions)
    #print(loaded_model.summary())

    rank_items = [random.randint(0, 30522) for _ in range(5)]
    rankings, probabilities = model.rank(example, rank_items, example["masked_lm_positions"])
    print(rank_items)
    print(rankings)
