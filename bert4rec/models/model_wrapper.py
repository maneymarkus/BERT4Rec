import abc
import tensorflow as tf
from typing import Union


class ModelWrapper(abc.ABC):
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self._meta_config = {
            "model": model.name,
            "tokenizer": None,
            "last_trained": None,
            "trained_on_dataset": None,
        }

    def get_meta(self) -> dict:
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
