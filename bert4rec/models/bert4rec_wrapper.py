from absl import logging
import json
import pathlib
import tensorflow as tf
import tensorflow_models as tfm
from typing import Union

from bert4rec.models.components import networks
from bert4rec.models.bert4rec_model import BERT4RecModel
from bert4rec.models.model_wrapper import ModelWrapper
import bert4rec.models.model_utils as utils
from bert4rec import tokenizers
from bert4rec.trainers import optimizers, trainer_utils

_ENCODER_CONFIG_FILE_NAME = "encoder_config.json"
_META_CONFIG_FILE_NAME = "meta_config.json"
_TOKENIZER_VOCAB_FILE_NAME = "vocab.txt"
_MODEL_WEIGHTS_FILES_PREFIX = "model_weights"


class BERT4RecModelWrapper(ModelWrapper):
    def __init__(self, bert_model: BERT4RecModel):
        super().__init__(bert_model)
        self.bert_model = bert_model
        self.update_meta(
            {
                "model": "BERT4Rec",
                "encoder_config": self.bert_model.encoder.get_config()
            }
        )

    def save(self, save_path: pathlib.Path, tokenizer: tokenizers.BaseTokenizer = None, mode: int = 0) -> True:
        """
        Saves a model to the disk. If the tokenizer is given, saves the used vocab from the tokenizer as well.

        NOTE: The model can only be saved successfully, when completely initialized. See model doc (further above)
        for more information.

        :param save_path: The path to save the model to
        :param tokenizer: If given, saves the used vocab from this tokenizer in the model directory
        :param mode: The mode determines where the model is stored. Available modes are 0, 1, 2.
        0 is default and stores the model relative to the project root. 1 stores the model relative
        to the virtual environment and 2 stores it relative to the current working directory. Modes 0 and 1
        also use predefined directories (saved_models). See `utils.determine_model_path()` for more info
        :return: True
        """
        save_path = utils.determine_model_path(save_path, mode)

        if self.bert_model.compiled_loss is None:
            raise RuntimeError("The model can't be saved without a loss. The model needs to be compiled first.")

        if self.bert_model.compiled_metrics and not self.bert_model.compiled_metrics._built:
            raise RuntimeError("The model can't be saved yet, as it is not fully instantiated and will "
                               "throw an error during saving. See model docs for more information.")

        logging.info(f"Saving {self.bert_model} to {save_path}")

        self.bert_model.save(save_path)

        if tokenizer:
            tokenizer.export_vocab_to_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
            self.update_meta({"tokenizer": tokenizer.identifier})

        # save meta config to file
        with open(save_path.joinpath(_META_CONFIG_FILE_NAME), "w") as f:
            json.dump(self._meta_config, f, indent=4)

        return True

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

        if not save_path.exists():
            raise ValueError(f"The given path {save_path} does not exist.")

        logging.info(f"Loading model from {save_path}")

        loaded_assets = dict()
        loaded_bert = tf.keras.models.load_model(save_path, custom_objects={
            "BERT4RecModel": BERT4RecModel,
            "Bert4RecEncoder": networks.Bert4RecEncoder,
            "OnDeviceEmbedding": tfm.nlp.layers.OnDeviceEmbedding,
            "PositionEmbedding": tfm.nlp.layers.PositionEmbedding,
            "SelfAttentionMask": tfm.nlp.layers.SelfAttentionMask,
            "TransformerEncoderBlock": tfm.nlp.layers.TransformerEncoderBlock,
            "RelativePositionEmbedding": tfm.nlp.layers.RelativePositionEmbedding,
            "RelativePositionBias": tfm.nlp.layers.RelativePositionBias,
            "AdamWeightDecay": optimizers.AdamWeightDecay,
            "masked_accuracy": trainer_utils.masked_accuracy,
            "MaskedSparseCategoricalCrossentropy": trainer_utils.MaskedSparseCategoricalCrossentropy,
        })

        wrapper = cls(loaded_bert)
        loaded_assets["model_wrapper"] = wrapper

        try:
            with open(save_path.joinpath(_META_CONFIG_FILE_NAME)) as jf:
                meta_config = json.load(jf)
                wrapper._meta_config = meta_config

                if "tokenizer" in meta_config and meta_config["tokenizer"] is not None:
                    tokenizer = tokenizers.get(meta_config["tokenizer"])
                    tokenizer.import_vocab_from_file(save_path.joinpath(_TOKENIZER_VOCAB_FILE_NAME))
                    loaded_assets["tokenizer"] = tokenizer

        except FileNotFoundError:
            logging.error(f"The meta configuration/information json file ({_META_CONFIG_FILE_NAME}) could "
                          f"not be found in the supposed model directory: {save_path}")

        return loaded_assets

    def rank_with_mlm_logits(self,
                             encoder_input: dict,
                             rank_items: list = None):

        encoder_output = self.bert_model(encoder_input, training=False)
        mlm_logits_batch = encoder_output["mlm_logits"]

        if "masked_lm_weights" in encoder_input:
            masked_lm_weights = tf.cast(encoder_input["masked_lm_weights"], tf.bool)
            mlm_logits_batch = tf.ragged.boolean_mask(mlm_logits_batch, masked_lm_weights)

        rankings = list()
        # iterate over batch
        for b_i, mlm_logits in enumerate(mlm_logits_batch):
            batch_rankings = []

            # iterate over tokens in this tensor
            for mlm_i, token_logits in enumerate(mlm_logits):
                if rank_items is not None and type(rank_items[0]) is list:
                    # => individual ranking list for each token (output)
                    rank_items_list = rank_items[b_i][mlm_i]
                    token_logits = tf.gather(token_logits, rank_items_list)
                    sorted_indexes = tf.argsort(token_logits, direction="DESCENDING")
                    ranking = tf.gather(rank_items_list, sorted_indexes)
                else:
                    ranking = tf.argsort(token_logits, direction="DESCENDING")

                batch_rankings.append(ranking)
            rankings.append(batch_rankings)
        return rankings

    def rank(self,
             encoder_input: dict,
             rank_items: list = None,
             sequence_indexes: Union[list, tf.Tensor] = None) -> tuple:
        """
        Ranks a given set of potential items (rank_items) by probability based on the given sequence.
        The items to be ranked should actually be the indexes of the items to gather their embeddings
        from the embeddings table, so basically just their numeric tokens.

        :param encoder_input: The input to the encoder on which the ranking of the rank_items is based on.
        Should be the regular BERT input dictionary
        :param rank_items: Either list of indexes of the items which should be ranked (in case
        sequence_indexes param is not given) or a list of lists of lists (shape: [batch, tokens, rank_items])
        with the indexes of the ranking items (if sequence_indexes param is given). If this is a
        list of lists of lists, the length of the primary list should be equal to the length of the
        sequence_indexes as each index in either list corresponds to the index of the other list.
        If none is given, the whole vocabulary (whole embedding table) is ranked.
        :param sequence_indexes: A list of integers determining the output positions
        which should be used to (individually) rank the items (based on the corresponding output
        of the sequence_output from the encoder). If none is given, the items are ranked based on
        the accumulated encoder output (pooled_output). Could legitimately be the masked language model
        (or simply the positions of the masked tokens which equals the 'masked_lm_positions' tensor).
        :return: A tuple with the first value containing the ranked items from highest probability to lowest
        and the second value containing the probabilities (of each item) (not sorted! -> can be cross-referenced
        with the original rank_items list)
        """
        gathered_embeddings = self.bert_model.encoder.get_embedding_table()

        if rank_items is not None and type(rank_items[0]) is not list:
            gathered_embeddings = tf.gather(self.bert_model.encoder.get_embedding_table(), rank_items)

        encoder_output = self.bert_model(encoder_input)

        probabilities = list()
        rankings = list()
        if sequence_indexes is not None:
            sequence_output = encoder_output["sequence_output"]
            # iterate over batch
            for b in range(len(sequence_indexes)):
                batch_rankings = []
                batch_probabilities = []

                # iterate over tokens in this tensor
                for i, token_index in enumerate(sequence_indexes[b]):
                    rank_items_list = rank_items
                    if rank_items is not None and type(rank_items[0]) is list:
                        # => individual ranking list for each token (output)
                        gathered_embeddings = \
                            tf.gather(self.bert_model.encoder.get_embedding_table(), rank_items[b][i])
                        rank_items_list = rank_items[b][i]
                    token_logits = sequence_output[b, token_index, :]
                    vocab_probabilities, ranking = utils.rank_items(token_logits, gathered_embeddings, rank_items_list)
                    batch_probabilities.append(vocab_probabilities)
                    batch_rankings.append(ranking)
                rankings.append(batch_rankings)
                probabilities.append(batch_probabilities)

        else:
            output_logits = encoder_output["pooled_output"][0, :]
            vocab_probabilities, ranking = utils.rank_items(output_logits, gathered_embeddings, rank_items)
            probabilities.append(vocab_probabilities)
            rankings.append(ranking)

        return rankings, probabilities
