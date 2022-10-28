from .base_tokenizer import *
from .simple_tokenizer import *


def get(identifier: str, **kwargs):
    if identifier == "simple":
        return SimpleTokenizer(**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
