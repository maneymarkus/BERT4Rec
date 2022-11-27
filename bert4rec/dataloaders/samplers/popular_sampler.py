from .base_sampler import BaseSampler

from bert4rec.dataloaders import dataloader_utils


class PopularSampler(BaseSampler):
    def __init__(self,
                 source: list = None,
                 sample_size: int = None):
        super().__init__(source, sample_size)

    def sample(self,
               source: list = None,
               sample_size: int = None,
               without: list = None) -> list:
        source, sample_size = self._get_parameters(source, sample_size)

        _source = source.copy()

        # remove elements from without from source
        if without is not None:
            _source = [i for i in _source if i not in without]

        _source = dataloader_utils.rank_items_by_popularity(_source)

        return _source[:sample_size]
