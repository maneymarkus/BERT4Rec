import numpy as np

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    """
    Samples randomly from a given source assuming a uniform distribution. If the vocab argument
    is given then this list will be used for sampling. If the vocab argument is not given
    but the source argument is then the duplicates will be removed from this list and the result
    will be set as the vocab list.

    """
    def __init__(self,
                 source: list = None,
                 vocab: list = None,
                 sample_size: int = None,
                 allow_duplicates: bool = False,
                 seed: int = None):
        super().__init__(source, vocab, sample_size)
        if self.vocab is None and self.source is not None:
            # remove duplicates by converting to set and then back to list
            self.vocab = list(set(self.source))
        self.allow_duplicates = allow_duplicates
        self.seed = seed

    def is_fully_prepared(self) -> bool:
        if self.vocab is None:
            return False
        if self.sample_size is None:
            return False
        return True

    def _get_parameters(self,
                        source: list = None,
                        vocab: list = None,
                        sample_size: int = None,
                        allow_duplicates: bool = None,
                        seed: int = None):
        source, vocab, sample_size = super()._get_parameters(source, vocab, sample_size)

        if vocab is None and source is not None and self.source is None:
            vocab = list(set(source))

        if vocab is None:
            raise ValueError("No vocab or any other source has been given to the random sampler.")

        if seed is None:
            seed = self.seed
        np.random.seed(seed)

        if allow_duplicates is None:
            allow_duplicates = self.allow_duplicates

        if allow_duplicates is False and sample_size > len(vocab):
            raise ValueError("When no duplicates are allowed in the final sample then the "
                             f"sample size (given sample size: {sample_size})) can not be greater "
                             "than the length length of the vocab (length of the vocab: "
                             f"{len(vocab)})")

        return source, vocab, sample_size, allow_duplicates

    def sample(self,
               sample_size: int = None,
               source: list = None,
               vocab: list = None,
               allow_duplicates: bool = None,
               seed: int = None,
               without: list = None) -> list:
        source, vocab, sample_size, allow_duplicates = \
            self._get_parameters(source, vocab, sample_size, allow_duplicates, seed)

        _source = vocab.copy()

        # remove elements from without from source
        if without is not None:
            _source = [i for i in _source if i not in without]

        return np.random.choice(_source, size=sample_size, replace=allow_duplicates).tolist()

    def set_source(self, source: list):
        if not self.allow_duplicates:
            # remove duplicates by converting to set and then back to list
            source = list(set(source.copy()))
        super().set_source(source)
