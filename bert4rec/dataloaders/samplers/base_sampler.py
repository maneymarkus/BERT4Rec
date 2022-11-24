import abc


class BaseSampler(abc.ABC):
    def __init__(self, ds: list = None, sample_size: int = None):
        if sample_size is not None and sample_size < 0:
            raise ValueError("The sample size shouldn't be negative to avoid unexpected outputs "
                             f"(Given: {sample_size})")
        if ds is not None:
            self.ds = ds.copy()
        self.sample_size = sample_size

    def _get_parameters(self,
                        ds: list = None,
                        sample_size: int = None):
        if ds is None:
            ds = self.ds
            if self.ds is None:
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

        return ds, sample_size

    @abc.abstractmethod
    def sample(self,
               ds: list = None,
               sample_size: int = None) -> list:
        pass

    def set_ds(self, ds: list):
        self.ds = ds

    def set_sample_size(self, sample_size: int):
        self.sample_size = sample_size
