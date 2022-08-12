from absl import logging
import collections
import copy
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


class BERTModel(tf.keras.Model):
    # TODO: save and reload model correctly (i.e. make functions from encoder class available again after reloading)
    def __init__(self,
                 encoder: networks.BertEncoder,
                 name: str = "bert",
                 **kwargs):

        super(BERTModel, self).__init__()

        inputs = copy.copy(encoder.inputs)
        predictions = encoder(inputs)

        self._config = {
            "encoder": encoder,
            "name": name,
        }

        self.encoder = encoder
        self.inputs = inputs

    def call(self, inputs, training: bool = False):
        if isinstance(inputs, list):
            logging.warning('List inputs to the Bert Model are discouraged.')
            inputs = dict([
                (ref.name, tensor) for ref, tensor in zip(self.inputs, inputs)
            ])

        outputs = dict()
        encoder_network_outputs = self.encoder(inputs)
        if isinstance(encoder_network_outputs, list):
            outputs['pooled_output'] = encoder_network_outputs[1]
            # When `encoder_network` was instantiated with return_all_encoder_outputs
            # set to True, `encoder_network_outputs[0]` is a list containing
            # all transformer layers' output.
            if isinstance(encoder_network_outputs[0], list):
                outputs['encoder_outputs'] = encoder_network_outputs[0]
                outputs['sequence_output'] = encoder_network_outputs[0][-1]
            else:
                outputs['sequence_output'] = encoder_network_outputs[0]
        elif isinstance(encoder_network_outputs, dict):
            outputs = encoder_network_outputs
        else:
            raise ValueError('encoder_network\'s output should be either a list '
                             'or a dict, but got %s' % encoder_network_outputs)

        return outputs

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)


class BERT4RecModelWrapper:
    def __init__(self,
                 bert_model: BERTModel):
        self.bert_model = bert_model
        self._meta_config = dict(
            {
                "model": "BERT4Rec",
                "tokenizer": None,
                "last_trained": None,
                "trained_on_dataset": None
            }
        )

    @classmethod
    def load(cls, save_path: pathlib.Path, mode: int = 0) -> dict:
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
        loaded_bert = tf.keras.models.load_model(save_path, custom_objects={"BERTModel": BERTModel})
        wrapper = cls(loaded_bert)
        loaded_assets["model"] = wrapper

        try:
            with open(save_path.joinpath(_META_CONFIG_FILE_NAME)) as jf:
                meta_config = json.load(jf)
                wrapper._meta_config = meta_config

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

        :param save_path: The path to save the model to
        :param tokenizer: If given, saves the used vocab from this tokenizer in the model directory
        :param mode: The mode determines where the model is stored. Available modes are 0, 1, 2.
        0 is default and stores the model relative to the project root. 1 stores the model relative
        to the virtual environment and 2 stores it relative to the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: True
        """
        save_path = utils.determine_model_path(save_path, mode)
        self.bert_model.save(save_path)

        if tokenizer:
            tokenizer.export_vocab_to_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
            self.update_meta({"tokenizer": tokenizer.code})

        with open(save_path.joinpath(_META_CONFIG_FILE_NAME), "w") as f:
            json.dump(self._meta_config, f, indent=4)

        return True

    def rank(self,
             encoder_input: dict,
             rank_items: list,
             sequence_indexes: Union[list, tf.Tensor] = None) -> tuple:
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
        and the second value containing the probabilities (of each item) (not sorted! -> can be cross referenced
        with the original rank_items list)
        """
        # TODO: save and reload model correctly (i.e. reloading the model into the right class again to make
        # functions available (as e.g. encoder.get_embedding_table()))
        gathered_embeddings = tf.gather(self.bert_model.encoder._embedding_layer.embeddings, rank_items)

        encoder_output = self.bert_model(encoder_input)

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

    def get_meta(self) -> dict:
        return self._meta_config

    def update_meta(self, updated_info: dict) -> True:
        self._meta_config.update(updated_info)
        return True

    def delete_keys_from_meta(self, keys: Union[list, str]) -> True:
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self._meta_config.pop(key, None)

        return True


if __name__ == "__main__":
    dataloader = BERT4RecML1MDataloader()
    dataloader.generate_vocab()
    ds = dataloader.preprocess_dataset()
    for element in ds.take(1):
        example = element
    print(example)
    tokenizer = dataloader.get_tokenizer()

    bert_encoder = networks.BertEncoder(30522)
    model = BERTModel(bert_encoder)
    model.compile(optimizer="adam", loss="mse")
    _ = model(model.inputs)
    wrapper = BERT4RecModelWrapper(model)
    embedding_table = model.encoder._embedding_layer.embeddings
    gathered_embeddings = tf.gather(embedding_table, [0, 3, 30519, 30521])

    wrapper.save(pathlib.Path("my_model"), tokenizer)
    loaded_assets = BERT4RecModelWrapper.load(pathlib.Path("my_model"))
    loaded_tokenizer = loaded_assets["tokenizer"]
    loaded_wrapper = loaded_assets["model"]

    predictions = model(example)
    print(predictions)

    rank_items = [random.randint(0, 30522) for _ in range(5)]
    rankings, probabilities = wrapper.rank(example, rank_items, example["masked_lm_positions"])
    print(rank_items)
    print(rankings)
