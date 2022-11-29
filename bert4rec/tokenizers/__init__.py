from typing import Union

from .base_tokenizer import BaseTokenizer
from .simple_tokenizer import SimpleTokenizer


tokenizers_map = {
    "simple": SimpleTokenizer
}


def get(identifier: Union[str, BaseTokenizer] = "simple", **kwargs) -> BaseTokenizer:
    """
    Factory method to return a concrete tokenizer instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if isinstance(identifier, str) and identifier in tokenizers_map:
        return tokenizers_map[identifier](**kwargs)
    elif isinstance(identifier, BaseTokenizer):
        return identifier
    else:
        raise ValueError(f"{identifier} is not known!")
