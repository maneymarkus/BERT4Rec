"""
Evaluation code and calculations taken from
https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run.py#L176
"""

import abc
import numpy as np


class EvaluationMetric(abc.ABC):
    def __init__(self, name: str, initial_value: int = 0):
        self._name = name
        self._initial_value = initial_value
        self._value = initial_value

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def update(self, rank: int):
        pass

    def reset(self):
        self._value = self._initial_value

    def result(self):
        return self._value


class RatioEvaluationMetric(EvaluationMetric):
    def __init__(self, name: str, initial_value: int = 0):
        super().__init__(name, initial_value)
        self._nominator = 0.0
        self._denominator = 0.0

    def update(self, rank: int):
        self._value = self._nominator / self._denominator
        return self._value

    def reset(self):
        super().reset()
        self._nominator = 0.0
        self._denominator = 0.0


class Counter(EvaluationMetric):
    """
    Simple counter metric that counts the amount of times the update method is called.
    May e.g. be used to keep track of the amount of evaluation steps
    """
    def __init__(self, name: str = "Counter", initial_value: int = 0):
        super().__init__(name, initial_value)

    def update(self, rank: int):
        self._value += 1


class HitRatio(RatioEvaluationMetric):
    def __init__(self, k: int, name: str = "HitRatio", initial_value: int = 0):
        name = name + "@" + str(k)
        super().__init__(name, initial_value)
        self._k = k

    def update(self, rank: int):
        self._denominator += 1
        if rank <= self._k:
            self._nominator += 1
        super().update(rank)


class NormalizedDiscountedCumulativeGain(RatioEvaluationMetric):
    def __init__(self, k: int, name: str = "NormalizedDiscountedCumulativeGain", initial_value: int = 0):
        name = name + "@" + str(k)
        super().__init__(name, initial_value)
        self._k = k

    def update(self, rank: int):
        self._denominator += 1
        if rank <= self._k:
            if rank == 1:
                _add = 1
            else:
                _add = 1 / np.log2(rank + 2)
            self._nominator += _add
        super().update(rank)


class MeanAveragePrecision(RatioEvaluationMetric):
    def __init__(self, name: str = "MeanAveragePrecision", initial_value: int = 0):
        super().__init__(name, initial_value)

    def update(self, rank: int):
        self._denominator += 1
        self._nominator += 1 / rank
        super().update(rank)


# Aliasing classes for shorter reference
class HR(HitRatio):
    def __init__(self, k: int, name: str = "HR", initial_value: int = 0):
        super().__init__(k, name, initial_value)


class NDCG(NormalizedDiscountedCumulativeGain):
    def __init__(self, k: int, name: str = "NDCG", initial_value: int = 0):
        super().__init__(k, name, initial_value)


class MAP(MeanAveragePrecision):
    def __init__(self, name: str = "MAP", initial_value: int = 0):
        super().__init__(name, initial_value)
