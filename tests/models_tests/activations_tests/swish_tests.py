"""Tests for the customized Swish activation."""
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from bert4rec.model.components import activations


@keras_parameterized.run_all_keras_modes
class CustomizedSwishTest(keras_parameterized.TestCase):

    def _hard_swish_np(self, x):
        x = np.float32(x)
        return x * np.clip(x + 3, 0, 6) / 6

    def test_simple_swish(self):
        features = [[.25, 0, -.25], [-1, -2, 3]]
        customized_swish_data = activations.simple_swish(features)
        swish_data = tf.nn.swish(features)
        self.assertAllClose(customized_swish_data, swish_data)

    def test_hard_swish(self):
        features = [[.25, 0, -.25], [-1, -2, 3]]
        customized_swish_data = activations.hard_swish(features)
        swish_data = self._hard_swish_np(features)
        self.assertAllClose(customized_swish_data, swish_data)


if __name__ == '__main__':
    tf.test.main()
