from .base_tokenizer import *
from .simple_tokenizer import *


class TokenizerFactory:
    def get_tokenizer(self, method: str, **kwargs):
        if method == "simple":
            return SimpleTokenizer(**kwargs)
        else:
            raise ValueError(f"{method} is not known!")


tokenizer_factory = TokenizerFactory()
