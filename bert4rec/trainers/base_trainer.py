import abc
import pathlib
import tensorflow as tf


class BaseTrainer(abc.ABC):
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.loss = None
        self.metrics = []
        self.callbacks = []

    @abc.abstractmethod
    def initialize_model(self,
                         optimizer: tf.keras.optimizers.Optimizer = None,
                         loss: tf.keras.losses.Loss = None,
                         metrics: list = None):
        """
        Initializes the model for training (basically compiles the model) and sets optimizer
        and loss function

        :return:
        """
        pass

    @abc.abstractmethod
    def train(self,
              train_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset,
              checkpoint_path: pathlib.Path,
              epochs: int = 50):
        pass

    @abc.abstractmethod
    def validate(self):
        pass
