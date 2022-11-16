"""Customized Sigmoid activation."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def hard_sigmoid(features):
    """Computes the hard sigmoid activation function.
    Args:
      features: A `Tensor` representing preactivation values.
    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return tf.nn.relu6(features + tf.cast(3., features.dtype)) * 0.16667
