import abc


class BaseSampler(abc.ABC):
    def __init__(self, source: list = None, sample_size: int = None):
        if sample_size is not None and sample_size < 0:
            raise ValueError("The sample size shouldn't be negative to avoid unexpected outputs "
                             f"(Given: {sample_size})")
        if source is not None:
            source = source.copy()
        self.source = source
        self.sample_size = sample_size

    def _get_parameters(self,
                        source: list = None,
                        sample_size: int = None):
        if source is None:
            source = self.source
            if self.source is None:
                raise ValueError("The dataset has to be given either during the initialization of the "
                                 "sampler or as an argument in the function call.")

        if sample_size is None:
            sample_size = self.sample_size
            if self.sample_size is None:
                raise ValueError("The sample size has to be given either during the initialization of the "
                                 "sampler or as an argument in the function call.")

        if sample_size < 0:
            raise ValueError("The sample size shouldn't be negative to avoid unexpected outputs "
                             f"(Given: {sample_size})")

        return source, sample_size

    @abc.abstractmethod
    def sample(self,
               source: list = None,
               sample_size: int = None,
               without: list = None) -> list:
        pass

    def set_source(self, source: list):
        self.source = source

    def set_sample_size(self, sample_size: int):
        self.sample_size = sample_size
