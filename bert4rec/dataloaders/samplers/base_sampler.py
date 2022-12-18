import abc
from absl import logging


class BaseSampler(abc.ABC):
    """
    Samplers may be initialized with a source list and a vocab list, which will most likely
    speed up processing. The `vocab` list contains the whole available (unique) vocab that may be
    sampled from. The `source` list should contain the occurrence of items in a dataset and
    therefore contains duplicates (might e.g. be necessary to calculate a probability distribution
    or to order the vocab)

    """
    def __init__(self, source: list = None, vocab: list = None, sample_size: int = None):
        if sample_size is not None and sample_size < 0:
            raise ValueError("The sample size shouldn't be negative to avoid unexpected outputs "
                             f"(Given: {sample_size})")
        if source is not None:
            source = source.copy()
        self.source = source

        if vocab is not None:
            vocab = vocab.copy()
        self.vocab = vocab

        self.sample_size = sample_size

    def _get_parameters(self,
                        source: list = None,
                        vocab: list = None,
                        sample_size: int = None):
        if source is None:
            source = self.source

        if vocab is None:
            vocab = self.vocab

        if sample_size is None:
            sample_size = self.sample_size
            if self.sample_size is None:
                raise ValueError("The sample size has to be given either during the initialization of the "
                                 "sampler or as an argument in the sample() method call.")

        if sample_size < 0:
            raise ValueError(f"A negative sample size is not allowed (Given: {sample_size})")

        return source, vocab, sample_size

    @abc.abstractmethod
    def sample(self,
               sample_size: int = None,
               source: list = None,
               vocab: list = None,
               without: list = None) -> list:
        pass

    @abc.abstractmethod
    def is_fully_prepared(self) -> bool:
        """
        Checks if the sampler is sufficiently initialized to be able to work
        when no argument is given to the `sample()` method call. The name might be confusing
        as this check does not have to mean that every attribute has to be set (depends on
        the concrete sampler).

        :return:
        """
        pass

    def set_source(self, source: list):
        self.source = source.copy()

    def set_vocab(self, vocab: list):
        self.vocab = vocab.copy()

    def set_sample_size(self, sample_size: int):
        self.sample_size = sample_size
