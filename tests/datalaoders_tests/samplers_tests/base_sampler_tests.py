from absl import logging
import tensorflow as tf

from bert4rec.dataloaders import samplers


class BaseSamplerTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        super().tearDown()

    def test_sampler_factory_method(self):
        sampler1 = samplers.get("random")
        self.assertIsInstance(sampler1, samplers.RandomSampler)
        sampler2 = samplers.get()
        self.assertIsInstance(sampler2, samplers.RandomSampler)
        with self.assertRaises(ValueError):
            samplers.get("slkfha")


if __name__ == '__main__':
    tf.test.main()
