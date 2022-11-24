from bert4rec.dataloaders.samplers.base_sampler import BaseSampler
from bert4rec.dataloaders.samplers.random_sampler import RandomSampler
from bert4rec.dataloaders.samplers.popular_sampler import PopularSampler


samplers_map = {
    "random": RandomSampler,
    "popular": PopularSampler
}


def get(identifier: str = "random", **kwargs) -> BaseSampler:
    """
    Factory method to return a concrete sampler instance according to the given identifier

    :param identifier:
    :param kwargs:
    :return:
    """
    if identifier in samplers_map:
        return samplers_map[identifier](**kwargs)
    else:
        raise ValueError(f"{identifier} is not known!")
