import copy
import random


def remove_elements_from_list(source: list, remove: list):
    """
    Removes elements in `remove` from a given list `source`

    :param source:
    :param remove:
    :return:
    """
    s = copy.copy(source)
    for r in remove:
        if r in s:
            s.remove(r)
    return s


def sample_random_items_from_list(item_list: list, sample_size: int):
    """
    Samples `sample_size` items randomly from a given list `item_list`. If `sample_size` is greater than the
    length of `item_list` than the `item_list` is simply shuffled and returned

    :param item_list:
    :param sample_size:
    :return:
    """
    items = copy.copy(item_list)
    random.shuffle(items)

    if len(items) <= sample_size:
        return items

    random_items = [items.pop(0) for _ in range(sample_size)]
    return random_items
