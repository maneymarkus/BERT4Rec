from .base_sampler import BaseSampler

from bert4rec.dataloaders import dataloader_utils


class PopularSampler(BaseSampler):
    def __init__(self,
                 ds: list = None,
                 sample_size: int = None,
                 allow_duplicates: bool = True,
                 seed: int = None):
        super().__init__(ds, sample_size)
        self.allow_duplicates = allow_duplicates
        self.seed = seed

    def sample(self,
               ds: list = None,
               sample_size: int = None,
               without: list = None) -> list:
        ds, sample_size = self._get_parameters(ds, sample_size)

        _ds = ds.copy()

        # remove elements from without from source
        if without is not None:
            _ds = [i for i in _ds if i not in without]

        _ds = dataloader_utils.rank_items_by_popularity(_ds)

        return _ds[:sample_size]
