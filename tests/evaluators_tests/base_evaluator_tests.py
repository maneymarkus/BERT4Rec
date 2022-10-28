import json
import pathlib
import tempfile
import tensorflow as tf
import time

import bert4rec.evaluation as evaluation


class BaseEvaluatorTest(tf.test.TestCase):
    def setUp(self):
        super(BaseEvaluatorTest, self).setUp()
        # Testing base evaluator features with a concrete tokenizer implementation (as abstract classes
        # can't be instantiated)
        self.evaluator = evaluation.get()
        self.evaluator.metrics.update({
            "metric": "test"
        })

    def tearDown(self):
        pass

    def test_save_results(self):
        tmpdir = tempfile.TemporaryDirectory()
        save_path_1 = pathlib.Path(tmpdir.name)

        tmpfile = tempfile.NamedTemporaryFile()
        # close temporary file as the resource will be opened again in the utils method (otherwise a permission
        # error will be raised)
        tmpfile.close()
        save_path_2 = pathlib.Path(tmpfile.name)
        save_path_1 = self.evaluator.save_results(save_path_1)
        save_path_2 = self.evaluator.save_results(save_path_2)

        self.assertTrue(save_path_1.is_file(),
                        f"Saving the results to a file should result in an existing file. "
                        f"{save_path_1} does not exist.")
        self.assertTrue(save_path_1.is_file(),
                        f"Saving the results to a file should result in an existing file. "
                        f"{save_path_2} does not exist.")
        with open(save_path_1, "r") as f:
            metrics = json.load(f)
            self.assertEqual(self.evaluator.metrics, metrics,
                             f"The saved metrics should be equal to the current status of the metrics of the "
                             f"evaluator object ({self.evaluator.metrics}), but actually are: {metrics}.")

        with open(save_path_2, "r") as f:
            metrics = json.load(f)
            self.assertEqual(self.evaluator.metrics, metrics,
                             f"The saved metrics should be equal to the current status of the metrics of the "
                             f"evaluator object ({self.evaluator.metrics}), but actually are: {metrics}.")


if __name__ == '__main__':
    tf.test.main()
