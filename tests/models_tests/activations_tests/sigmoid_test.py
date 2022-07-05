"""Tests for the customized Sigmoid activation."""

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
# pylint: disable=g-direct-tensorflow-import
from bert4rec.model.components import activations


@keras_parameterized.run_all_keras_modes
class CustomizedSigmoidTest(keras_parameterized.TestCase):

    def _hard_sigmoid_nn(self, x):
        x = np.float32(x)
        return tf.nn.relu6(x + 3.) * 0.16667

    def test_hard_sigmoid(self):
        features = [[.25, 0, -.25], [-1, -2, 3]]
        customized_hard_sigmoid_data = activations.hard_sigmoid(features)
        sigmoid_data = self._hard_sigmoid_nn(features)
        self.assertAllClose(customized_hard_sigmoid_data, sigmoid_data)


if __name__ == '__main__':
    tf.test.main()
