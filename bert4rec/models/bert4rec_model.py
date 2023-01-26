from absl import logging
import copy
import tensorflow as tf
import tensorflow_models as tfm
from typing import Optional

from bert4rec.dataloaders.bert4rec_dataloader import BERT4RecDataloader
from bert4rec.models.components import networks

_ENCODER_CONFIG_FILE_NAME = "encoder_config.json"
_META_CONFIG_FILE_NAME = "meta_config.json"
_TOKENIZER_VOCAB_FILE_NAME = "vocab.txt"
_MODEL_WEIGHTS_FILES_PREFIX = "model_weights"

train_step_signature = [{
    "labels": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_word_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_positions": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "masked_lm_weights": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
}]

SPECIAL_TOKEN_IDS = BERT4RecDataloader(0, 0)._SPECIAL_TOKEN_IDS


class BERT4RecModel(tf.keras.Model):
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
                 encoder: networks.Bert4RecEncoder,
                 customized_masked_lm: Optional[tf.keras.layers.Layer] = None,
                 mlm_activation="gelu",
                 mlm_initializer="glorot_uniform",
                 name: str = "bert4rec",
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

        super().__init__(name=name, **kwargs)

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

        self.masked_lm = customized_masked_lm or tfm.nlp.layers.MaskedLM(
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
            #self.prediction_mask = tf.sparse.to_dense(sparse_mask)
            self.prediction_mask = None

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
            "input_mask": inputs["input_mask"],
        }
        encoder_network_outputs = self.encoder(encoder_inputs, training=training)
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
        config = super().get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)

    def rank_items(self,
                   encoder_input: dict,
                   items: list = None):
        """
        Ranks given `items` according to the encoder output (based on the masked_lm layer output).
        If `items` is given, which is a list of lists containing tokens, that represent the items
        to be ranked. Each list in that list should correspond to a mask item in this batch.
        Therefore, the amount of given item lists should be equal to the number of mask tokens.
        If `items` is None than the whole vocabulary is ranked.

        """

        encoder_output = self(encoder_input, training=False)
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
                if items is not None and type(items[0]) is list:
                    # => individual ranking list for each token (output)
                    rank_items_list = items[b_i][mlm_i]
                    token_logits = tf.gather(token_logits, rank_items_list)
                    sorted_indexes = tf.argsort(token_logits, direction="DESCENDING")
                    ranking = tf.gather(rank_items_list, sorted_indexes)
                else:
                    ranking = tf.argsort(token_logits, direction="DESCENDING")

                batch_rankings.append(ranking)
            rankings.append(batch_rankings)
        return rankings
