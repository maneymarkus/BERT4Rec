import abc
import datetime
import pathlib
import tensorflow as tf

from bert4rec.dataloaders.base_dataloader import BaseDataloader


class BaseTrainer(abc.ABC):
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.optimizer = None
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
              checkpoint_path: pathlib.Path = None,
              epochs: int = 50,
              steps_per_epoch: int = None,
              validation_steps: int = None):
        pass

    def update_wrapper_meta_info(self, wrapper, dataloader: BaseDataloader):
        wrapper.update_meta({
            "last_trained": str(datetime.datetime.now()),
            "trained_on_dataset": dataloader.dataset_identifier
        })

    @abc.abstractmethod
    def validate(self):
        pass

    def append_callback(self, callback: tf.keras.callbacks.Callback):
        self.callbacks.append(callback)
