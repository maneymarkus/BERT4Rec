import random

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self,
                 ds: list = None,
                 sample_size: int = None,
                 allow_duplicates: bool = True,
                 seed: int = None):
        super().__init__(ds, sample_size)
        self.allow_duplicates = allow_duplicates
        self.seed = seed

    def _get_parameters(self,
                        ds: list = None,
                        sample_size: int = None,
                        allow_duplicates: bool = True,
                        seed: int = None):
        ds, sample_size = super()._get_parameters(ds, sample_size)

        if seed is None:
            seed = self.seed
        random.seed(seed)

        if allow_duplicates is None:
            allow_duplicates = self.allow_duplicates
            if self.allow_duplicates is None:
                raise ValueError("The allow duplicates argument has to be given either during the "
                                 "initialization of the sampler or as an argument in the "
                                 "function call.")

        return ds, sample_size, allow_duplicates

    def sample(self,
               ds: list = None,
               sample_size: int = None,
               allow_duplicates: bool = True,
               seed: int = None,
               without: list = None) -> list:
        ds, sample_size, allow_duplicates = \
            self._get_parameters(ds, sample_size, allow_duplicates, seed)

        _ds = ds.copy()
        random.shuffle(_ds)
        # remove duplicates by converting to set and then back to list
        if not allow_duplicates:
            _ds = list(set(_ds))

        # remove elements from without from source
        if without is not None:
            _ds = [i for i in _ds if i not in without]

        return _ds[:sample_size]
