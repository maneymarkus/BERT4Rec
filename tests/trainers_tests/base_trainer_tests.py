from absl import logging
import tensorflow as tf

from bert4rec.models.components import networks
from bert4rec.models import BERT4RecModel
from bert4rec import trainers


class BaseTrainerTests(tf.test.TestCase):
    def setUp(self):
        super(BaseTrainerTests, self).setUp()
        logging.set_verbosity(logging.DEBUG)
        self.bert_encoder = networks.Bert4RecEncoder(200)
        self.bert_model = BERT4RecModel(self.bert_encoder)
        # initialize concrete trainer to test base (abstract) trainer features
        self.trainer = trainers.BERT4RecTrainer(self.bert_model)

    def tearDown(self):
        pass

    def test_trainer_factory_method(self):
        trainer = trainers.get("bert4rec", **{"model": self.bert_model})
        self.assertIsInstance(trainer, trainers.BERT4RecTrainer)
        trainer2 = trainers.get(**{"model": self.bert_model})
        self.assertIsInstance(trainer2, trainers.BERT4RecTrainer)
        with self.assertRaises(ValueError):
            trainers.get("alsdkjfhoiho")

    def test_append_callback(self):
        self.assertEqual(self.trainer.callbacks, [],
                         f"Prior to appending callbacks, the callbacks property should be an empty list, "
                         f"but is: {self.trainer.callbacks}")
        callback = tf.keras.callbacks.History()
        self.trainer.append_callback(callback)
        self.assertLen(self.trainer.callbacks, 1,
                       f"After appending a callback, the callbacks property should have exactly one element, "
                       f"but actually has: {len(self.trainer.callbacks)} elements.\n"
                       f"Callbacks property:{self.trainer.callbacks}")
        self.assertIsInstance(self.trainer.callbacks[0], tf.keras.callbacks.History)
        print(self.trainer.callbacks)


if __name__ == "__main__":
    tf.test.main()
