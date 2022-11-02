from .base_tokenizer import *
from .simple_tokenizer import *


tokenizers_map = {
    "simple": SimpleTokenizer
}


def get(identifier: str = "simple", **kwargs):
    """
    Factory method to return a concrete tokenizer instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if identifier in tokenizers_map:
        return tokenizers_map[identifier](**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
