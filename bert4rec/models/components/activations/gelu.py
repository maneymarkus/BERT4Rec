"""Gaussian error linear unit."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    return tf.keras.activations.gelu(x, approximate=True)
