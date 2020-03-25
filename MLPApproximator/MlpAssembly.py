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

    def __init__(self, input_number, output_number, hidden_layer_number=3):
        self.__input_number = input_number
        self.__output_number = output_number
        self.__p1 = Perceptron(input_number, hidden_layer_number)
        self.__p2 = Perceptron(hidden_layer_number, output_number, test_first_p1=False)

    def forwardPropagation(self, input_data):
        self.__p2.forwardPropagation(self.__p1.forwardPropagation(input_data).output())

    def doWeirdStuff(self, output_data):
        self__mean_square_error = self.__p2.meanSquaredError(output_data)
        print('P2: mean error \n', self__mean_square_error)


class MlpApproximatorTester:
    def run(self) -> str:
        # sample: x=(1,2); output: f(x) = (1, 0).
        first_sample = np.array([1, 2]).reshape((2, 1))
        expected_out = np.array([1, 0]).reshape((2, 1))

        mlp_approximator = MlpApproximator(2, 2)
        mlp_approximator.forwardPropagation(first_sample)
        mlp_approximator.doWeirdStuff(expected_out)

        return 0
