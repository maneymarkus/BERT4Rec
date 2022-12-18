from absl import logging
import numpy as np
import tqdm

from .base_sampler import BaseSampler


class PopularRandomSampler(BaseSampler):
    """
    This sampler is the combination of the popular and random samplers. For each item in
    a given `source` a probability is calculated on the basis of the popularity of each item
    in the given `source` (popularity is simply determined according to the frequency of
    occurrences). Then `sample_size` items are randomly sampled from the source given the
    calculated probability distribution.

    """

    def __init__(self,
                 source: list = None,
                 vocab: list = None,
                 sample_size: int = None,
                 allow_duplicates: bool = False,
                 seed: int = None):
        super().__init__(source, vocab, sample_size)
        # already rank source list for performance
        self.vocab = vocab
        # probability distribution will have the same length as vocab
        self.probability_distribution = []
        self.allow_duplicates = allow_duplicates
        self.seed = seed
        if self.source is not None and self.vocab is not None:
            self._determine_probability_distribution(self.source, self.vocab)

    def is_fully_prepared(self) -> bool:
        if self.vocab is None:
            return False
        if self.probability_distribution is []:
            return False
        if len(self.vocab) != len(self.probability_distribution):
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

        if seed is None:
            seed = self.seed
        np.random.seed(seed)

        if source is None:
            raise ValueError("The source argument has to be given either during the "
                             "initialization of the sampler or as an argument in the "
                             "sample method call when working with the popular random sampler.")

        if vocab is None:
            raise ValueError("The vocab argument has to be given either during the "
                             "initialization of the sampler or as an argument in the "
                             "sample method call when working with the popular random sampler.")

        if allow_duplicates is None:
            allow_duplicates = self.allow_duplicates

        if allow_duplicates is False and sample_size > len(vocab):
            raise ValueError("When no duplicates are allowed in the final sample then the "
                             f"sample size (given sample size: {sample_size})) can not be greater "
                             f"than the length of the vocab (length of the vocab: {len(vocab)})")

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

        # source list only has to be ranked if the sampler wasn't initialized with it or
        # the source was not set via the setter method
        if not self.probability_distribution:
            self._determine_probability_distribution(source, vocab)

        # sample more items than actually needed, maybe remove some elements and then return
        # sample size elements
        size = sample_size

        # TODO: Sampling from a list with a bigger sample size than the remaining list
        #  after removing `without` elements yields unexpected results

        # make sure there are no duplicates in without
        if without is not None:
            without = list(set(without))
            size += len(without)

        if not allow_duplicates and size > len(vocab):
            raise ValueError(f"The given without list (length: {len(without)} reduces the vocab "
                             f"(length: {len(vocab)}) too much to take a sample of size "
                             f"{sample_size} (since no duplicates are allowed).")

        sample = np.random.choice(vocab, size, allow_duplicates, self.probability_distribution).tolist()

        # remove elements from without from sample
        if without is not None:
            sample = [i for i in sample if i not in without]

        return sample[:sample_size]

    def _determine_probability_distribution(self, source: list, vocab: list):
        self.probability_distribution = []
        total_items = len(source)
        logging.info("Generate probability distribution for popular random sampler")
        for item in tqdm.tqdm(vocab):
            item_count = source.count(item)
            item_probability = item_count / total_items
            self.probability_distribution.append(item_probability)

    def set_source(self, source: list):
        super().set_source(source)
        if self.vocab is not None:
            self._determine_probability_distribution(self.source, self.vocab)

    def set_vocab(self, vocab: list):
        super().set_vocab(vocab)
        if self.source is not None:
            self._determine_probability_distribution(self.source, self.vocab)
