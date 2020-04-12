#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np


class SigmoidActivationFunction:
    """
    Threshold function for the output
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, val) -> float:
        return self.activate(val)

    def activate(self, val: np.array):
        return 1. / (1. + np.exp(-val))

    def differentiate(self, val: np.array) -> np.array:
        return self.activate(val) * (1. - self.activate(val))


class TanhActivationFunction:
    """
    Threshold function for the output
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, val) -> float:
        return self.__activate(val)

    def activate(self, val: np.array):
        return np.tanh(val)

    def differentiate(self, val: np.array) -> np.array:
        return 1 - self.activate(val) ** 2
