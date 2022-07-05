"""Tests for the Gaussian error linear unit."""

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from bert4rec.model.components import activations


@keras_parameterized.run_all_keras_modes
class GeluTest(keras_parameterized.TestCase):

    def test_gelu(self):
        expected_data = [[0.14967535, 0., -0.10032465],
                         [-0.15880796, -0.04540223, 2.9963627]]
        gelu_data = activations.gelu([[.25, 0, -.25], [-1, -2, 3]])
        self.assertAllClose(expected_data, gelu_data)


if __name__ == '__main__':
    tf.test.main()
