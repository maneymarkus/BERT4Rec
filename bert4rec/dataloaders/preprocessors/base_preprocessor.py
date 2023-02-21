import abc
import tensorflow as tf


class BasePreprocessor(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def set_properties(cls, **kwargs):
        """
        Set preprocessor dependant properties necessary for the processing/preparation of the data
        """
        pass

    @classmethod
    @abc.abstractmethod
    def process_element(cls, *args):
        """
        Processes a single element. Will often be used in combination with the process_dataset
        method and is therefore thought to be the map_func in the tf.data.Dataset.map(map_func)
        (See: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map). Arguments may vary
        then as the .map() method yields the content of dataset elements (e.g. the dataset consists
        of tuples with various values, then the values are yielded individually).
        """
        pass

    @classmethod
    @abc.abstractmethod
    def process_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Applies the process_element function to the whole dataset via the tf.data.Dataset.map()
        method
        """
        pass

    @classmethod
    @abc.abstractmethod
    def prepare_inference(cls, data):
        """
        Prepares given data for inference.
        """
        pass
