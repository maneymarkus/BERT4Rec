from absl import logging
import random
import tensorflow as tf

from bert4rec.evaluation import evaluation_utils as utils


class EvaluatorUtilsTests(tf.test.TestCase):
    def setUp(self):
        super(EvaluatorUtilsTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)

    def tearDown(self):
        pass

    def test_remove_elements_from_list(self):
        list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        remove_elements_1 = [1, 2, 3, 4, 5]
        expected_result_1 = [6, 7, 8, 9]
        actual_result_1 = utils.remove_elements_from_list(list_1, remove_elements_1)
        self.assertEqual(expected_result_1, actual_result_1,
                         f"Removing these elements: {remove_elements_1}\nfrom this list: {list_1} should yield "
                         f"this list: {expected_result_1}\nbut actually outputs: {actual_result_1}")

    def test_sample_random_items_from_list(self):
        sample_size = 10
        list_1 = [random.randint(0, 100) for _ in range(50)]
        list_2 = [random.randint(0, 100) for _ in range(10)]
        list_3 = [random.randint(0, 100) for _ in range(5)]

        sampled_list_1 = utils.sample_random_items_from_list(list_1, sample_size)
        sampled_list_2 = utils.sample_random_items_from_list(list_2, sample_size)
        sampled_list_3 = utils.sample_random_items_from_list(list_3, sample_size)

        for item in sampled_list_1:
            self.assertIn(item, list_1,
                          f"Each item in the sampled list: {sampled_list_1}\nshould also be in the original "
                          f"list ({list_1}). However, {item} is not in the original list.")
        self.assertSameElements(list_2, sampled_list_2)
        self.assertSameElements(list_3, sampled_list_3)

        self.assertLen(sampled_list_1, 10)
        self.assertLen(sampled_list_2, len(list_2))
        self.assertLen(sampled_list_3, len(list_3))


if __name__ == '__main__':
    tf.test.main()
