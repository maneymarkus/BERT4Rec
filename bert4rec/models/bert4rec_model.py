from absl import logging
import copy
import json
import pathlib
import tensorflow as tf
from typing import Union, Optional

from bert4rec.dataloaders import BERT4RecDataloader
from bert4rec.models.components import layers, networks
import bert4rec.models.model_utils as utils
import bert4rec.tokenizers as tokenizers
from bert4rec.trainers import optimizers, trainer_utils

_ENCODER_CONFIG_FILE_NAME = "encoder_config.json"
_META_CONFIG_FILE_NAME = "meta_config.json"
_TOKENIZER_VOCAB_FILE_NAME = "vocab.txt"
_MODEL_WEIGHTS_FILES_PREFIX = "model_weights"

train_step_signature = [{
    "labels": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_word_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_type_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_positions": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_weights": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
}]


SPECIAL_TOKEN_IDS = BERT4RecDataloader(0, 0)._SPECIAL_TOKEN_IDS


class BERTModel(tf.keras.Model):
    """
    NOTE: The model can only be saved, when completely initialized (when using the saving api).
    For a not further known reason (but empirically tested), saving a subclassed Keras model with a
    custom `train_step()` function throws an error when not fully initialized. In detail, this line
    `loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)` causes the error. Fully
    initialized means in this context that the given metrics and loss(es) in the `model.compile()` call are not
    built but only set/initialized. See e.g. `init__()` method of the LossesContainer object
    (wrapper for compiled_metrics property):
    https://github.com/keras-team/keras/blob/3cec735c5602a1bd9880b1b5735c5ce64a94eb76/keras/engine/compile_utils.py#L117
    """

    def __init__(self,
                 encoder: networks.BertEncoder,
                 customized_masked_lm: Optional[tf.keras.layers.Layer] = None,
                 mlm_activation=None,
                 mlm_initializer="glorot_uniform",
                 name: str = "bert",
                 special_token_ids: list[int] = SPECIAL_TOKEN_IDS,
                 **kwargs):
        """

        :param encoder:
        :param customized_masked_lm:
        :param mlm_activation:
        :param mlm_initializer:
        :param name: Name of this keras model
        :param special_token_ids: An optional list of special token ids that should be prevented from
            being predicted
        :param kwargs:
        """

        super(BERTModel, self).__init__(name=name, **kwargs)

        self._config = {
            "encoder": encoder,
            "customized_masked_lm": customized_masked_lm,
            "mlm_activation": mlm_activation,
            "mlm_initializer": mlm_initializer,
            "name": name,
        }

        self.encoder = encoder
        self.vocab_size = encoder.get_config()["vocab_size"]

        _ = self.encoder(self.encoder.inputs)

        inputs = copy.copy(encoder.inputs)

        self.masked_lm = customized_masked_lm or layers.MaskedLM(
            self.encoder.get_embedding_table(),
            activation=mlm_activation,
            initializer=mlm_initializer,
            name="cls/predictions"
        )

        masked_lm_positions = tf.keras.layers.Input(
            shape=(None,), name="masked_lm_positions", dtype=tf.int32
        )
        if isinstance(inputs, dict):
            inputs["masked_lm_positions"] = masked_lm_positions
        else:
            inputs.append(masked_lm_positions)

        # create prediction mask from special token ids to prevent these tokens from being predicted
        if special_token_ids is not None:
            # create a tensor of shape (#num_special_tokens, 1) from special token ids list
            skip_ids = tf.constant(special_token_ids, dtype=tf.int64)[:, None]
            sparse_mask = tf.SparseTensor(
                values=[-float("inf")] * len(skip_ids),
                indices=skip_ids,
                # match the shape (or simply the length) of the vocabulary
                dense_shape=[self.vocab_size]
            )
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        self.inputs = inputs

    @property
    def identifier(self):
        return "bert4rec"

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, list):
            logging.warning('List inputs to the Bert Model are discouraged.')
            inputs = dict([
                (ref.name, tensor) for ref, tensor in zip(self.inputs, inputs)
            ])

        outputs = dict()
        encoder_inputs = {
            "input_word_ids": inputs["input_word_ids"],
            "input_type_ids": inputs["input_type_ids"],
            "input_mask": inputs["input_mask"],
        }
        encoder_network_outputs = self.encoder(encoder_inputs)
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

        sequence_output = outputs["sequence_output"]
        # Inference may not have masked_lm_positions and mlm_logits are not needed
        if "masked_lm_positions" in inputs:
            masked_lm_positions = inputs["masked_lm_positions"]
            predicted_logits = self.masked_lm(sequence_output, masked_lm_positions)
            # apply the prediction mask if given
            if self.prediction_mask is not None:
                predicted_logits += self.prediction_mask
            outputs["mlm_logits"] = predicted_logits

        return outputs

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs):
        """
        Custom train_step function to alter standard training behaviour

        :return:
        """
        y_true = inputs["masked_lm_ids"]

        with tf.GradientTape() as tape:
            encoder_output = self(inputs, training=True)
            y_pred = encoder_output["mlm_logits"]

            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=train_step_signature)
    def test_step(self, inputs):
        """
        Custom train_step function to alter standard training behaviour

        :return:
        """
        y_true = inputs["masked_lm_ids"]

        encoder_output = self(inputs, training=False)
        # logits
        y_pred = encoder_output["mlm_logits"]

        loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super(BERTModel, self).get_config()
        config.update(self._config)
        return config

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
                "trained_on_dataset": None,
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

        loaded_assets = dict()
        loaded_bert = tf.keras.models.load_model(save_path, custom_objects={
            "BERTModel": BERTModel,
            "AdamWeightDecay": optimizers.AdamWeightDecay,
            "BertEncoderV2": networks.BertEncoder,
            "OnDeviceEmbedding": layers.OnDeviceEmbedding,
            "PositionEmbedding": layers.PositionEmbedding,
            "SelfAttentionMask": layers.SelfAttentionMask,
            "TransformerEncoderBlock": layers.TransformerEncoderBlock,
            "RelativePositionEmbedding": layers.RelativePositionEmbedding,
            "RelativePositionBias": layers.RelativePositionBias,
            "masked_accuracy": trainer_utils.masked_accuracy,
            "MaskedSparseCategoricalCrossentropy": trainer_utils.MaskedSparseCategoricalCrossentropy,
            "approx_gelu": networks.bert_encoder.approx_gelu
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

        encoder_output = self.bert_model(encoder_input)
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
                    token_logits = tf.gather(token_logits, rank_items[b_i][mlm_i], axis=-1)
                    rank_items_list = rank_items[b_i][mlm_i]
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
