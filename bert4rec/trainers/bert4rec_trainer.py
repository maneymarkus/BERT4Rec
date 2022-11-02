import pathlib
import tensorflow as tf

from bert4rec.trainers.base_trainer import BaseTrainer
from bert4rec.trainers import optimizers


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
            # Define loss function. Ignore class is set up to ignore "pad class". See:
            # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
        self.loss = loss

        if metrics is None:
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy()
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
                monitor="val_sparse_categorical_accuracy",
                save_best_only=True
            )
            self.append_callback(model_checkpoint_callback)
            if checkpoint_path.parent.is_dir():
                if tf.train.latest_checkpoint(checkpoint_path.parent) is not None:
                    self.model.load_weights(checkpoint_path)

        history = self.model.fit(x=train_ds,
                                 validation_data=val_ds,
                                 epochs=epochs,
                                 callbacks=self.callbacks)
        return history

    def validate(self):
        pass
