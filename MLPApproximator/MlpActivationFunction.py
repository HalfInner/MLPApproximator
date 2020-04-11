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

    def differentiate(self, diff: np.array, val: np.array) -> np.array:
        if diff.shape != val.shape:
            raise ValueError('Diff shape={} must be same as Val shape={}'.format(diff.shape, val.shape))
        act = self.activate(val)
        return diff * (act * (1. - act))


class ReluActivationFunction:
    """
    Threshold function for the output
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, val) -> float:
        return self.activate(val)

    def activate(self, val: np.array):
        return np.maximum(0., val)

    def differentiate(self, diff: np.array, val: np.array) -> np.array:
        if diff.shape != val.shape:
            raise ValueError('Diff shape={} must be same as Val shape={}'.format(diff.shape, val.shape))
        out = np.copy(diff * val)
        return np.where(out <= 0, 0, 1)

