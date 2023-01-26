import abc
import tensorflow as tf
from typing import Union


class ModelWrapper(abc.ABC):
    """
    The model wrapper class is supposed to provide additional (mostly administrative
    like extended saving and loading functionality) functionality for the wrapped model.
    It basically depicts a custom decorator.
    The custom_objects class variable corresponds to the custom_objects argument in the
    TensorFlow Keras load_model() method
    (See: https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model#args_1)
    """

    _model: tf.keras.Model = None
    _custom_objects: dict = {}

    def __init__(self, model: tf.keras.Model):
        self._model = model
        self._meta_config = {
            "model": model.name,
            "tokenizer": None,
            "last_trained": None,
            "trained_on_dataset": None,
        }

    @property
    def model(self):
        return self._model

    def get_meta_config(self) -> dict:
        return self._meta_config

    def update_meta(self, updated_info: dict) -> True:
        self._meta_config.update(updated_info)
        return True

    def delete_keys_from_meta(self, keys: Union[list, str]) -> True:
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self._meta_config.pop(key, None)

        return True
