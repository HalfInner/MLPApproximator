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

    def activate(self, val):
        return 1. / (1. + np.exp(-val))

    def differentiate(self, diff, val):
        act = self.activate(val)
        return diff * act * (1 - act)


class TanhActivationFunction:
    """
    Threshold function for the output
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, val) -> float:
        return self.__activate(val)

    def __activate(self, val):
        return np.tanh(val)
