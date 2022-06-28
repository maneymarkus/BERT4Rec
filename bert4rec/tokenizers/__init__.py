from .base_tokenizer import *
from .simple_tokenizer import *


class TokenizerFactory:
    def get_tokenizer(self, method: str, **kwargs):
        if method == "simple":
            return SimpleTokenizer(**kwargs)
        else:
            raise ValueError(method)


tokenizer_factory = TokenizerFactory()
