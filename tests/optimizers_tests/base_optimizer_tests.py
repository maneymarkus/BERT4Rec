from absl import logging
import tensorflow as tf

from bert4rec.trainers import optimizers


class GeneralOptimizerTests(tf.test.TestCase):
    def setUp(self):
        super(GeneralOptimizerTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        pass

    def test_optimizer_factory_method(self):
        optimizer = optimizers.get("adamw")
        self.assertIsInstance(optimizer, optimizers.AdamWeightDecay)
        optimizer2 = optimizers.get()
        self.assertIsInstance(optimizer2, optimizers.AdamWeightDecay)
        with self.assertRaises(ValueError):
            optimizers.get("alsdkjfhoiho")


if __name__ == "__main__":
    tf.test.main()
