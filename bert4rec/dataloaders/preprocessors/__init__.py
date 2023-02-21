"""
Preprocessors do the data preprocessing. A dataset is given and the unprocessed data
gets processed to prepare it for inputting it into a ml model
"""
from typing import Union

from .base_preprocessor import BasePreprocessor
from .bert4rec_preprocessor import BERT4RecPreprocessor
from .bert4rec_temporal_preprocessor import BERT4RecTemporalPreprocessor

preprocessors_map = {
    "bert4rec": BERT4RecPreprocessor,
    "bert4rec_temporal": BERT4RecTemporalPreprocessor,
}


def get(identifier: Union[str, BasePreprocessor] = "popular", **kwargs) -> BasePreprocessor:
    """
    Factory method to return a concrete sampler instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if isinstance(identifier, str) and identifier in preprocessors_map:
        return preprocessors_map[identifier](**kwargs)
    elif isinstance(identifier, BasePreprocessor):
        return identifier
    else:
        raise ValueError(f"{identifier} is not known!")
