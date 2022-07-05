"""Customized Swish activation."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def simple_swish(features):
    """Computes the Swish activation function.
    The tf.nn.swish operation uses a custom gradient to reduce memory usage.
    Since saving custom gradients in SavedModel is currently not supported, and
    one would not be able to use an exported TF-Hub module for fine-tuning, we
    provide this wrapper that can allow to select whether to use the native
    TensorFlow swish operation, or whether to use a customized operation that
    has uses default TensorFlow gradient computation.
    Args:
      features: A `Tensor` representing preactivation values.
    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return features * tf.nn.sigmoid(features)


@tf.keras.utils.register_keras_serializable(package='Text')
def hard_swish(features):
    """Computes a hard version of the swish function.
    This operation can be used to reduce computational cost and improve
    quantization for edge devices.
    Args:
      features: A `Tensor` representing preactivation values.
    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    fdtype = features.dtype
    return features * tf.nn.relu6(features + tf.cast(3., fdtype)) * (1. / 6.)


@tf.keras.utils.register_keras_serializable(package='Text')
def identity(features):
    """Computes the identity function.
    Useful for helping in quantization.
    Args:
      features: A `Tensor` representing preactivation values.
    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return tf.identity(features)
