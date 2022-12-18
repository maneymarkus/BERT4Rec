from absl import logging
import tensorflow as tf

from bert4rec.dataloaders import samplers
from tests import test_utils


class PopularRandomSamplerTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        logging.set_verbosity(logging.DEBUG)
        self.sampler = samplers.PopularRandomSampler()

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
        vocab = list(set(ds))

        # sampling should not change the original ds
        ds_copy = ds.copy()
        _ = self.sampler.sample(5, ds, vocab)
        self.assertListEqual(ds, ds_copy)

        # set source and vocab in sampler
        self.sampler.set_vocab(vocab)
        self.sampler.set_source(ds)

        sample_size = 20
        sample_1 = self.sampler.sample(sample_size)
        self._assert_sample(sample_1, ds, sample_size)

        sample_size = 220
        sample_2 = self.sampler.sample(sample_size, allow_duplicates=True)
        self._assert_sample(sample_2, ds, sample_size)

        with self.assertRaises(ValueError):
            self.sampler.sample(-3, ds)
            self.sampler.sample(3, None)
            self.sampler.sample(220, ds, allow_duplicates=False)

        # initialize sampler with initial values
        sampler_config = {
            "source": ds,
            "vocab": vocab,
            "sample_size": 20
        }
        self.sampler = samplers.PopularRandomSampler(**sampler_config)

        # test sample without certain elements and less than sample_size remaining elements
        without_list = ds[:-2]
        with self.assertRaises(ValueError):
            self.sampler.sample(without=without_list, allow_duplicates=False)

        # TODO
        #sample_5 = self.sampler.sample(without=without_list, allow_duplicates=True)
        #for sample_element in sample_5:
        #    self.assertNotIn(sample_element, without_list)
        #expected_length = 20
        #self._assert_sample(sample_5, ds, expected_length)

        # test sample without certain elements and more than sample_size remaining elements
        without_list = ds[:-50]
        sample_6 = self.sampler.sample(without=without_list)
        for sample_element in sample_6:
            self.assertNotIn(sample_element, without_list)
        expected_length = 20
        self._assert_sample(sample_6, ds, expected_length)


if __name__ == '__main__':
    tf.test.main()
