#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np

from MLPApproximator.MlpActivationFunction import SigmoidActivationFunction


class Perceptron:
    """
    Perceptron must be controlled by upper layer
    """

    def __init__(self, input_number, output_number, activation_function=SigmoidActivationFunction(),
                 debug_on=False, weight=None) -> None:
        self.__debug_on = debug_on

        self.__input_number = input_number
        self.__output_number = output_number
        self.__activation_function = activation_function

        if weight is None:
            self.__weights = np.ones((output_number, input_number), dtype=float)
        else:
            self.__weights = weight

        self.__weights = self.__weights * 0.1
        required_shape = (output_number, input_number)
        if self.__weights.shape != required_shape:
            raise ValueError('Dimension of weights must meet requirements of input and output Expect={} Actual={}'
                             .format(self.__weights.shape, required_shape))
        self.__correction = None
        self.__delta_weights = None

        self.__learning_ratio = 1.2

        self.__output_data = np.zeros(shape=[3, 1])
        self.__mean_squared_error = None
        self.__input_data = None

    def train(self):
        """
        Train the perceptron.
        """
        if self.__input_data is None:
            raise RuntimeError('Cannot proceed train without input')
        self.__weights += self.__delta_weights
        self.__debug('Weights \n', self.__weights)

    def weights(self):
        """
        Might be converted to private

        :return:
        """
        return self.__weights

    def forwardPropagation(self, input_data):
        """
        Might be converted to private

        :param input_data:
        :return:
        """
        self.__input_data = input_data
        self.__debug('Input=\n{}\n\tWeights=\n{}'.format(input_data, self.__weights))

        raw_output = self.__weights @ self.__input_data
        self.__debug('Raw Out=\n{}'.format(raw_output))

        self.__output_data = self.__activation_function(raw_output)
        self.__debug('Activated Out=\n{}'.format(self.__output_data))

        return self.__output_data

    def propagateBackward(self, expected_out):
        """
        Might be converted to private

        :param expected_out:
        :return:
        """
        if not expected_out.shape == self.__output_data.shape:
            raise ValueError("Shape of validator must be same as the output")

        self.__debug('ExpectedOut=\n{}'.format(expected_out))
        self.__debug('Out=\n{}'.format(self.__output_data))
        step1 = expected_out - self.__output_data
        self.__calculateCorrectionAndWeights(step1)

        return self.__correction, self.__weights

    def propagateHiddenBackward(self, next_correction, next_weight):
        """
        Might be converted to private
        :param next_correction:
        :param next_weight:
        :return:
        """
        # # sys.exit(-1)
        # self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), 1 - np.power(self.A1, 2))
        # self.dW1 = (1 / m) * np.dot(self.dZ1, X)
        # self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)

        step1 = next_weight.transpose() @ next_correction
        self.__calculateCorrectionAndWeights(step1)

        return self.__correction, self.__weights

    def output(self) -> np.array:
        """
        Might be converted to private
        :return:
        """
        return self.__output_data

    def __calculateCorrectionAndWeights(self, step1):
        step2 = np.ones_like(self.__output_data) - self.__output_data
        self.__correction = step1 * step2 * self.__output_data
        self.__debug('correction=\n{}'.format(self.__correction))
        self.__delta_weights = self.__learning_ratio * self.__correction @ self.__input_data.transpose()
        self.__debug('Delta weights=\n{}'.format(self.__delta_weights))
        self.__weights = self.__weights + self.__delta_weights
        self.__debug('New Weights=\n{}'.format(self.__weights))

    def __debug(self, msg, *args):
        if self.__debug_on:
            print('Perceptron: \n\t', msg, *args)
