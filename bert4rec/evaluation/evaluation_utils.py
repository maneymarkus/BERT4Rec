import copy


def remove_elements_from_list(source: list, remove: list):
    s = copy.copy(source)
    for r in remove:
        if r in s:
            s.remove(r)
    return s
