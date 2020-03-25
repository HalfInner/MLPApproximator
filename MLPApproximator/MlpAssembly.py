from __future__ import absolute_import, division, print_function, unicode_literals

import sys
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

    def __init__(self, input_number, output_number, activation_function=SigmoidActivationFunction(),
                 test_first_p1=True) -> None:
        self.__input_number = input_number
        self.__output_number = output_number
        self.__activation_function = activation_function

        self.__weights = np.ones((output_number, input_number), dtype=float)
        if test_first_p1:
            self.__weights[0][1] = -1
            self.__weights[2][0] = -1
        else:
            self.__weights[0][1] = -1
            self.__weights[1][0] = -1
            self.__weights[1][2] = -1

        self.__output_data = np.zeros_like(self.__weights)

    def forwardPropagation(self, input_data):
        raw_output = self.__weights @ input_data
        self.__output_data = self.__activation_function(raw_output)

        return self

    def meanSquaredError(self, expected_out):
        if not expected_out.shape == self.__output_data.shape:
            raise ValueError("Shape of validator must be same as the output")
        el1 = expected_out - self.__output_data
        el2 = np.ones([self.__output_number, 1]) - self.__output_data
        el3 = self.__output_data

        mean_squared_error = el1 * el2 * el3
        return mean_squared_error

    def backwardPropagation(self, validate_out):
        pass

    def output(self) -> np.array:
        return self.__output_data


class MlpApproximator:
    def run(self) -> str:
        # sample: x=(1,2); output: f(x) = (1, 0).
        p1 = Perceptron(2, 3)
        p2 = Perceptron(3, 2, test_first_p1=False)

        first_sample = np.array([1, 2]).reshape((2, 1))
        p1_out = p1.forwardPropagation(first_sample).output()
        p2.forwardPropagation(p1_out)

        print('Out1: \n', p1.output())
        print('Out2: \n', p2.output())

        expected_out = np.array([1, 0]).reshape((2, 1))
        p2_error = p2.meanSquaredError(expected_out)
        print('P2: mean error \n', p2_error)

        return 0
