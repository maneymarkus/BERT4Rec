"""Tests for the customized Relu activation."""

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
# pylint: disable=g-direct-tensorflow-import
from bert4rec.model.components import activations


@keras_parameterized.run_all_keras_modes
class CustomizedReluTest(keras_parameterized.TestCase):

    def test_relu6(self):
        features = [[.25, 0, -.25], [-1, -2, 3]]
        customized_relu6_data = activations.relu6(features)
        relu6_data = tf.nn.relu6(features)
        self.assertAllClose(customized_relu6_data, relu6_data)


if __name__ == '__main__':
    tf.test.main()
