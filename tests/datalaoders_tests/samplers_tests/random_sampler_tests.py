from absl import logging
import tensorflow as tf
import random

from bert4rec.dataloaders import samplers
from tests import test_utils


class RandomSamplerTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.sampler = samplers.get("random")

    def tearDown(self):
        super().tearDown()

    def _assert_sample(self, sample: list, ds: list, expected_sample_size: int):
        self.assertLen(sample, expected_sample_size)
        ds_set = set(ds)
        self.assertAllInSet(sample, ds_set)

    def test_sample(self):
        seed = None
        ds_size = 200

        ds = test_utils.generate_random_word_list(size=ds_size, seed=seed)

        # sampling should not change the original ds
        ds_copy = ds.copy()
        _ = self.sampler.sample(ds, 5)
        self.assertListEqual(ds, ds_copy)

        sample_size = 20
        sample_1 = self.sampler.sample(ds, sample_size)
        self._assert_sample(sample_1, ds, sample_size)

        sample_size = 220
        sample_2 = self.sampler.sample(ds, sample_size)
        self._assert_sample(sample_2, ds, len(ds))

        sample_size = -3
        with self.assertRaises(ValueError):
            self.sampler.sample(ds, sample_size)
            self.sampler.sample(None, 3)

        sample_size = 5
        seed = 319
        sample_3 = self.sampler.sample(ds, sample_size, seed=seed)
        seed = 5015
        sample_4 = self.sampler.sample(ds, sample_size, seed=seed)
        self.assertNotAllEqual(sample_3, sample_4)

        # initialize sampler with initial values
        sampler_config = {
            "ds": ds,
            "sample_size": 20
        }
        self.sampler = samplers.get("random", **sampler_config)

        # test sample without certain elements and less than sample_size remaining elements
        without_list = ds[:-2]
        sample_5 = self.sampler.sample(without=without_list)
        for sample_element in sample_5:
            self.assertNotIn(sample_element, without_list)
        expected_length = len(ds) - len(without_list)
        self._assert_sample(sample_5, ds, expected_length)

        # test sample without certain elements and more than sample_size remaining elements
        without_list = ds[:-50]
        sample_6 = self.sampler.sample(without=without_list)
        for sample_element in sample_6:
            self.assertNotIn(sample_element, without_list)
        expected_length = 20
        self._assert_sample(sample_6, ds, expected_length)


if __name__ == '__main__':
    tf.test.main()
