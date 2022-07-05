"""Customized Relu activation."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def relu6(features):
    """Computes the Relu6 activation function.
    Args:
      features: A `Tensor` representing preactivation values.
    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return tf.nn.relu6(features)
