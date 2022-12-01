import random

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self,
                 source: list = None,
                 sample_size: int = None,
                 allow_duplicates: bool = True,
                 seed: int = None):
        if source and not allow_duplicates:
            # remove duplicates by converting to set and then back to list
            source = list(set(source.copy()))
        super().__init__(source, sample_size)
        self.allow_duplicates = allow_duplicates
        self.seed = seed

    def _get_parameters(self,
                        source: list = None,
                        sample_size: int = None,
                        allow_duplicates: bool = True,
                        seed: int = None):
        source, sample_size = super()._get_parameters(source, sample_size)

        if seed is None:
            seed = self.seed
        random.seed(seed)

        if allow_duplicates is None:
            allow_duplicates = self.allow_duplicates
            if self.allow_duplicates is None:
                raise ValueError("The allow duplicates argument has to be given either during the "
                                 "initialization of the sampler or as an argument in the "
                                 "function call.")

        return source, sample_size, allow_duplicates

    def sample(self,
               source: list = None,
               sample_size: int = None,
               allow_duplicates: bool = True,
               seed: int = None,
               without: list = None) -> list:
        source, sample_size, allow_duplicates = \
            self._get_parameters(source, sample_size, allow_duplicates, seed)

        _source = source.copy()
        random.shuffle(_source)
        # remove duplicates by converting to set and then back to list
        if not allow_duplicates:
            _source = list(set(_source))

        # remove elements from without from source
        if without is not None:
            _source = [i for i in _source if i not in without]

        return _source[:sample_size]

    def set_source(self, source: list):
        if not self.allow_duplicates:
            # remove duplicates by converting to set and then back to list
            source = list(set(source.copy()))
        super().set_source(source)
