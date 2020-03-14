from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import time

import numpy as np


class SigmoidActivationFunction:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, val) -> float:
        return self.__activate(val)

    def __activate(self, val):
        return 1. / (1. + np.exp(-val))


class Perceptron:

    def __init__(self, input_number, output_number, activation_function=SigmoidActivationFunction()) -> None:
        self.__input_number = input_number
        self.__output_number = output_number
        self.__activation_function = activation_function

        self.__weights = np.ones([input_number, output_number], dtype=float).reshape([3, 2])
        self.__output_data = np.zeros_like(self.__weights)

    def forwardPropagation(self, input_data) -> np.array:
        raw_output = self.__weights @ input_data
        for el in raw_output:
            el = self.__activation_function(el)
        self.__output_data = raw_output

    def output(self) -> np.array:
        return self.__output_data


class MlpApproximator:
    def run(self) -> str:
        p1 = Perceptron(2, 3)
        p1.forwardPropagation(np.array([1, 2]).reshape((2, 1)))
        print('Out: \n', p1.output())
        return 0
