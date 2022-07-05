"""Tests for tf_utils."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations

from bert4rec.model.components import model_utils


def all_strategy_combinations():
    return combinations.combine(
        strategy=[
            strategy_combinations.cloud_tpu_strategy,
            strategy_combinations.mirrored_strategy_with_two_gpus,
        ],
        mode='eager',
    )


class TFUtilsTest(tf.test.TestCase, parameterized.TestCase):

    @combinations.generate(all_strategy_combinations())
    def test_cross_replica_concat(self, strategy):
        num_cores = strategy.num_replicas_in_sync

        shape = (2, 3, 4)

        def concat(axis):

            @tf.function
            def function():
                replica_value = tf.fill(shape, model_utils.get_replica_id())
                return model_utils.cross_replica_concat(replica_value, axis=axis)

            return function

        def expected(axis):
            values = [np.full(shape, i) for i in range(num_cores)]
            return np.concatenate(values, axis=axis)

        per_replica_results = strategy.run(concat(axis=0))
        replica_0_result = per_replica_results.values[0].numpy()
        for value in per_replica_results.values[1:]:
            self.assertAllClose(value.numpy(), replica_0_result)
        self.assertAllClose(replica_0_result, expected(axis=0))

        replica_0_result = strategy.run(concat(axis=1)).values[0].numpy()
        self.assertAllClose(replica_0_result, expected(axis=1))

        replica_0_result = strategy.run(concat(axis=2)).values[0].numpy()
        self.assertAllClose(replica_0_result, expected(axis=2))

    @combinations.generate(all_strategy_combinations())
    def test_cross_replica_concat_gradient(self, strategy):
        num_cores = strategy.num_replicas_in_sync

        shape = (10, 5)

        @tf.function
        def function():
            replica_value = tf.random.normal(shape)
            with tf.GradientTape() as tape:
                tape.watch(replica_value)
                concat_value = model_utils.cross_replica_concat(replica_value, axis=0)
                output = tf.reduce_sum(concat_value)
            return tape.gradient(output, replica_value)

        per_replica_gradients = strategy.run(function)
        for gradient in per_replica_gradients.values:
            self.assertAllClose(gradient, num_cores * tf.ones(shape))


if __name__ == '__main__':
    tf.test.main()
