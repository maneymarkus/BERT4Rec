from .base_sampler import BaseSampler
from bert4rec.dataloaders import dataloader_utils


class PopularSampler(BaseSampler):
    """
    Samples "rigidly" according to the popularity of items in given sources.
    Popularity is being determined on the basis of the frequency of occurrences of individual
    items. Then the items are sorted according to their popularity and the sample method returns
    the first `sample_size` items from this sorted list.

    """
    def __init__(self,
                 source: list = None,
                 sample_size: int = None):
        super().__init__(source, sample_size)
        # already rank source list for performance
        if self.source is not None:
            self.source = dataloader_utils.rank_items_by_popularity(self.source)

    def sample(self,
               sample_size: int = None,
               source: list = None,
               without: list = None) -> list:
        source, sample_size = self._get_parameters(source, sample_size)

        _source = source.copy()

        # remove elements from without from source
        if without is not None:
            _source = [i for i in _source if i not in without]

        # source list only has to be ranked if the sampler wasn't initialized with it or
        # the source was not set via the setter method
        if self.source is None:
            _source = dataloader_utils.rank_items_by_popularity(_source)

        return _source[:sample_size]

    def set_source(self, source: list):
        super().set_source(source)
        self.source = dataloader_utils.rank_items_by_popularity(self.source)
