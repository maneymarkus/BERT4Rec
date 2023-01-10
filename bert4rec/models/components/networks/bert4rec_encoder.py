"""Transformer-based BERT encoder network."""
# pylint: disable=g-classes-have-attributes

from typing import Any, Callable, Optional, Union
from absl import logging
import tensorflow as tf
import tensorflow_models as tfm

_Initializer = Union[str, tf.keras.initializers.Initializer]
_Activation = Union[str, Callable[..., Any]]


class Bert4RecEncoder(tf.keras.layers.Layer):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT4Rec: Sequential Recommendation with Bidirectional Encoder
    Representations from Transformer" (https://arxiv.org/abs/1810.04805).
    It includes the embedding lookups (but no type embeddings) and transformer
    layers, but not the masked language model or classification task networks.

    The default values for this object are taken from the BERT-Base implementation
    in "BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding".

    Args:
      vocab_size: The size of the token vocabulary.
      hidden_size: The size of the transformer hidden layers.
      num_layers: The number of transformer layers.
      num_attention_heads: The number of attention heads for each transformer. The
        hidden size must be divisible by the number of attention heads.
      max_sequence_length: The maximum sequence length that this encoder can
        consume. If None, max_sequence_length uses the value from sequence length.
        This determines the variable shape for positional embeddings.
      type_vocab_size: The number of types that the 'type_ids' input can take.
      inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network for each transformer.
      inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network for each transformer.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: The dropout rate to use for the attention layers within
        the transformer layers.
      initializer: The initialzer to use for all weights in this encoder.
      output_range: The sequence output range, [0, output_range), by slicing the
        target sequence of the last transformer layer. `None` means the entire
        target sequence will attend to the source sequence, which yields the full
        output.
      embedding_width: The width of the word embeddings. If the embedding width is
        not equal to hidden size, embedding parameters will be factorized into two
        matrices in the shape of ['vocab_size', 'embedding_width'] and
        ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
        smaller than 'hidden_size').
      embedding_layer: An optional Layer instance which will be called to generate
        embeddings for the input word IDs.
      norm_first: Whether to normalize inputs to attention and intermediate dense
        layers. If set False, output of attention and intermediate dense layers is
        normalized.
      with_dense_inputs: Whether to accept dense embeddings as the input.
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_layers: int = 12,
            num_attention_heads: int = 12,
            max_sequence_length: int = 512,
            inner_dim: int = 3072,
            inner_activation: _Activation = "gelu",
            output_dropout: float = 0.1,
            attention_dropout: float = 0.1,
            initializer: _Initializer = tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            output_range: Optional[int] = None,
            embedding_width: Optional[int] = None,
            embedding_layer: Optional[tf.keras.layers.Layer] = None,
            norm_first: bool = False,
            with_dense_inputs: bool = False,
            **kwargs):
        super().__init__(**kwargs)

        activation = tf.keras.activations.get(inner_activation)
        initializer = tf.keras.initializers.get(initializer)

        if embedding_width is None:
            embedding_width = hidden_size

        if embedding_layer is None:
            self._embedding_layer = tfm.nlp.layers.OnDeviceEmbedding(
                vocab_size=vocab_size,
                embedding_width=embedding_width,
                initializer=tfm.utils.clone_initializer(initializer),
                name='word_embeddings')
        else:
            self._embedding_layer = embedding_layer

        self._position_embedding_layer = tfm.nlp.layers.PositionEmbedding(
            initializer=tfm.utils.clone_initializer(initializer),
            max_length=max_sequence_length,
            name='position_embedding')

        self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

        self._embedding_dropout = tf.keras.layers.Dropout(
            rate=output_dropout, name='embedding_dropout')

        # We project the 'embedding' output to 'hidden_size' if it is not already
        # 'hidden_size'.
        self._embedding_projection = None
        if embedding_width != hidden_size:
            self._embedding_projection = tf.keras.layers.EinsumDense(
                '...x,xy->...y',
                output_shape=hidden_size,
                bias_axes='y',
                kernel_initializer=tfm.utils.clone_initializer(initializer),
                name='embedding_projection')

        self._transformer_layers = []
        self._attention_mask_layer = tfm.nlp.layers.SelfAttentionMask(
            name='self_attention_mask')
        for i in range(num_layers):
            layer = tfm.nlp.layers.TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_dim=inner_dim,
                inner_activation=activation,
                output_dropout=output_dropout,
                attention_dropout=attention_dropout,
                norm_first=norm_first,
                output_range=output_range if i == num_layers - 1 else None,
                kernel_initializer=tfm.utils.clone_initializer(initializer),
                name='transformer/layer_%d' % i)
            self._transformer_layers.append(layer)

        self._pooler_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            kernel_initializer=tfm.utils.clone_initializer(initializer),
            name='pooler_transform')

        self._config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'max_sequence_length': max_sequence_length,
            'inner_dim': inner_dim,
            'inner_activation': tf.keras.activations.serialize(activation),
            'output_dropout': output_dropout,
            'attention_dropout': attention_dropout,
            'initializer': tf.keras.initializers.serialize(initializer),
            'output_range': output_range,
            'embedding_width': embedding_width,
            'embedding_layer': embedding_layer,
            'norm_first': norm_first,
            'with_dense_inputs': with_dense_inputs,
        }
        if with_dense_inputs:
            self.inputs = dict(
                input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
                input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
                dense_inputs=tf.keras.Input(
                    shape=(None, embedding_width), dtype=tf.float32),
                dense_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
            )
        else:
            self.inputs = dict(
                input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
                input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32)
            )

    def call(self, inputs):
        word_embeddings = None
        if isinstance(inputs, dict):
            word_ids = inputs.get('input_word_ids')
            mask = inputs.get('input_mask')
            word_embeddings = inputs.get('input_word_embeddings', None)

            dense_inputs = inputs.get('dense_inputs', None)
            dense_mask = inputs.get('dense_mask', None)
        else:
            raise ValueError('Unexpected inputs type to %s.' % self.__class__)

        if word_embeddings is None:
            word_embeddings = self._embedding_layer(word_ids)

        if dense_inputs is not None:
            # Concat the dense embeddings at sequence end.
            word_embeddings = tf.concat([word_embeddings, dense_inputs], axis=1)
            mask = tf.concat([mask, dense_mask], axis=1)

        # absolute position embeddings.
        position_embeddings = self._position_embedding_layer(word_embeddings)

        embeddings = word_embeddings + position_embeddings
        embeddings = self._embedding_norm_layer(embeddings)
        embeddings = self._embedding_dropout(embeddings)

        if self._embedding_projection is not None:
            embeddings = self._embedding_projection(embeddings)

        attention_mask = self._attention_mask_layer(embeddings, mask)

        encoder_outputs = []
        x = embeddings
        for layer in self._transformer_layers:
            x = layer([x, attention_mask])
            encoder_outputs.append(x)

        last_encoder_output = encoder_outputs[-1]
        first_token_tensor = last_encoder_output[:, 0, :]
        pooled_output = self._pooler_layer(first_token_tensor)

        return dict(
            sequence_output=encoder_outputs[-1],
            pooled_output=pooled_output,
            encoder_outputs=encoder_outputs)

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_embedding_layer(self):
        return self._embedding_layer

    def get_config(self):
        return dict(self._config)

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @property
    def pooler_layer(self):
        """The pooler dense layer after the transformer layers."""
        return self._pooler_layer

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'embedding_layer' in config and config['embedding_layer'] is not None:
            warn_string = (
                'You are reloading a model that was saved with a '
                'potentially-shared embedding layer object. If you continue to '
                'train this model, the embedding layer will no longer be shared. '
                'To work around this, load the model outside of the Keras API.')
            print('WARNING: ' + warn_string)
            logging.warning(warn_string)

        return cls(**config)
