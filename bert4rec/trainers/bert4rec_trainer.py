from absl import logging
import pathlib
import tensorflow as tf

from bert4rec.dataloaders import BaseDataloader, dataloader_utils
from bert4rec.trainers.base_trainer import BaseTrainer
from bert4rec.trainers import optimizers, trainer_utils


class BERT4RecTrainer(BaseTrainer):
    def __init__(self, model: tf.keras.Model):
        super(BERT4RecTrainer, self).__init__(model)
        
    def initialize_model(self,
                         optimizer: tf.keras.optimizers.Optimizer = None,
                         loss: tf.keras.losses.Loss = None,
                         metrics: list = None):

        if optimizer is None:
            # use default values if no other optimizer is provided
            optimizer = optimizers.get("adamw")
        self.optimizer = optimizer

        if loss is None:
            loss = trainer_utils.MaskedSparseCategoricalCrossentropy()
        self.loss = loss

        if metrics is None:
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(),
                trainer_utils.masked_accuracy
            ]
        self.metrics = metrics

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self,
              train_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset,
              checkpoint_path: pathlib.Path = None,
              epochs: int = 50):

        if checkpoint_path:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor="val_masked_accuracy",
                save_best_only=True
            )
            self.append_callback(model_checkpoint_callback)
            if checkpoint_path.parent.is_dir() and tf.train.latest_checkpoint(checkpoint_path.parent) is not None:
                self.model.load_weights(checkpoint_path)

        history = self.model.fit(x=train_ds,
                                 validation_data=val_ds,
                                 epochs=epochs,
                                 callbacks=self.callbacks)
        return history

    def smart_training(self,
                       dataloader: BaseDataloader,
                       checkpoint_path: pathlib.Path = None,
                       epochs: int = 500,
                       early_stopping_patience: int = 3,
                       make_batches_params: dict = {}) -> list:
        """
        Smart training will initialize an EarlyStoppingCallback. Then it trains the given model either until
        the epochs have been run through or until the early stopping callback executed. If the latter is the
        case then it will preprocess the training data again (to have the masked lm applied differently) and restart
        training with the remaining epochs.

        :param dataloader:
        :param checkpoint_path:
        :param epochs:
        :param early_stopping_patience:
        :param make_batches_params:
        :return:
        """

        if make_batches_params is None:
            make_batches_params = {}
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
        self.append_callback(early_stopping_callback)

        histories = []

        era = 1
        while epochs > 0:
            print(f"Era {era}:")
            # prepare training data (again, if epochs are left still)
            train_ds, val_ds, test_ds = dataloader.prepare_training()
            train_batches = dataloader_utils.make_batches(train_ds, **make_batches_params)
            val_batches = dataloader_utils.make_batches(val_ds, **make_batches_params)

            # train the model
            history = self.train(train_batches, val_batches, checkpoint_path, epochs)
            histories.append(history)

            # get trained epochs via loss attribute of the history object (each epoch has a loss)
            trained_epochs = len(history["loss"])
            epochs -= trained_epochs
            era += 1

        return histories

    def train_extended(self,
                       dataloader: BaseDataloader,
                       checkpoint_path: pathlib.Path = None,
                       epochs: int = 25,
                       eras: int = 25,
                       make_batches_params: dict = {}) -> list:
        """


        :param dataloader:
        :param checkpoint_path:
        :param epochs:
        :param eras:
        :param make_batches_params:
        :return:
        """

        histories = []

        for era in range(eras):
            print(f"Era {era}/{eras}:")
            # prepare training data (again, if epochs are left still)
            train_ds, val_ds, test_ds = dataloader.prepare_training()
            train_batches = dataloader_utils.make_batches(train_ds, **make_batches_params)
            val_batches = dataloader_utils.make_batches(val_ds, **make_batches_params)

            # train the model
            history = self.train(train_batches, val_batches, checkpoint_path, epochs)
            histories.append(history)

        return histories

    def validate(self):
        pass
