#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np


class SigmoidActivationFunction:
    """
    Threshold function for the output
    """

    def activate(self, val: np.array):
        return 1. / (1. + np.exp(-val))

    def differentiate(self, val: np.array) -> np.array:
        return self.activate(val) * (1. - self.activate(val))


class TanhActivationFunction:
    """
    Threshold function for the output
    """

    def activate(self, val: np.array):
        return np.tanh(val)

    def differentiate(self, val: np.array) -> np.array:
        return 1 - self.activate(val) ** 2


class LinearActivationFunction:
    """
    Linear function for the output
    """

    def activate(self, val: np.array):
        return val

    def differentiate(self, val: np.array) -> np.array:
        return 1


class ReLUActivationFunction:
    """
    Threshold function for the output
    """

    def activate(self, val: np.array):
        val[val < 0.] = 0.
        return

    def differentiate(self, val: np.array) -> np.array:
        val[val < 0] = 0
        val[val >= 0] = 0
        return val
