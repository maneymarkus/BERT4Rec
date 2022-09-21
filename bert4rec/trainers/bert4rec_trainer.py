import pathlib
import tensorflow as tf

from bert4rec.trainers.base_trainer import BaseTrainer
from bert4rec.trainers.optimizers import get_optimizer_factory


class BERT4RecTrainer(BaseTrainer):
    def __init__(self, model: tf.keras.Model):
        super(BERT4RecTrainer, self).__init__(model)
        
    def initialize_model(self,
                         optimizer: tf.keras.optimizers.Optimizer = None,
                         loss: tf.keras.losses.Loss = None,
                         metrics: list = None):

        if optimizer is None:
            optimizer_factory = get_optimizer_factory("bert4rec")
            # use default values if no other optimizer is provided
            optimizer = optimizer_factory.create_adam_w_optimizer()
        self.optimizer = optimizer

        if loss is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
            if checkpoint_path.exists():
                self.model.load_weights(checkpoint_path)

        history = self.model.fit(x=train_ds,
                                 validation_data=val_ds,
                                 epochs=epochs,
                                 callbacks=self.callbacks)
        return history

    def validate(self):
        pass