from absl import logging

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
                 vocab: list = None,
                 sample_size: int = None):
        super().__init__(source, vocab, sample_size)
        # already rank source list for performance
        if self.source is not None:
            self.source = dataloader_utils.rank_items_by_popularity(self.source)

    def is_fully_prepared(self) -> bool:
        if self.source is None:
            return False
        if self.sample_size is None:
            return False
        return True

    def _get_parameters(self,
                        source: list = None,
                        vocab: list = None,
                        sample_size: int = None):
        source, vocab, sample_size = super()._get_parameters(source, vocab, sample_size)

        if vocab is not None:
            logging.info("The vocab argument is not necessary for the popular sampler. It just "
                         "supports it for compatibility reasons.")

        if source is None:
            raise ValueError("The source argument has to be provided to the popular sampler but "
                             "None was given.")

        if sample_size >= len(source):
            logging.info(f"The given sample size ({sample_size}) is bigger than the length of "
                         "the given source list. The popular sampler will then return just "
                         "the whole source list and the sample size will be smaller than "
                         "wanted.")

        return source, vocab, sample_size

    def sample(self,
               sample_size: int = None,
               source: list = None,
               vocab: list = None,
               without: list = None) -> list:
        source, vocab, sample_size = self._get_parameters(source, vocab, sample_size)

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
