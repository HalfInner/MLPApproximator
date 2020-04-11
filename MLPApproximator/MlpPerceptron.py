#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np

from MLPApproximator.MlpActivationFunction import SigmoidActivationFunction


class Perceptron:
    """
    Perceptron must be controlled by upper layer
    """

    def __init__(self, input_number, output_number, activation_function=SigmoidActivationFunction(),
                 debug_on=False, weight=None, name="Perceptron") -> None:
        self.__debug_on = debug_on

        self.__input_number = input_number
        self.__output_number = output_number
        self.__activation_function = activation_function
        self.__name = name

        self.__debug(' input number={}'.format(input_number))
        self.__debug('output number={}'.format(output_number))

        if weight is None:
            self.__weights = np.random.randn(output_number, input_number) * 0.5
        else:
            self.__weights = weight

        required_shape = (output_number, input_number)
        self.__delta_weights = None

        self.__learning_ratio = 1

        self.__raw_output = np.zeros(shape=[3, 1])
        self.__output_data = np.zeros_like(self.__raw_output)
        self.__mean_squared_error = None
        self.__input_data = None

    def forwardPropagation(self, input_data):
        """
        Might be converted to private

        :param input_data:
        :return:
        """
        self.__input_data = input_data
        self.__debug('Input=\n{}'.format(input_data))
        self.__debug('Weights=\n{}'.format(self.__weights))

        # self.__raw_output = self.__input_data.dot(self.__weights.T)
        self.__raw_output = self.__weights.dot(self.__input_data.T).T
        self.__debug('Raw Out=\n{}'.format(self.__raw_output))

        self.__output_data = self.__activation_function.activate(self.__raw_output)
        self.__debug('Activated Out=\n{}'.format(self.__output_data))

        return self.__output_data

    def propagateBackward(self, expected_out):
        """
        Might be converted to private

        :param expected_out:
        :return:
        """
        # if not expected_out.shape == self.__output_data.shape:
        #     raise ValueError("Shape of validator must be same as the output")

        self.__debug('ExpectedOut=\n{}'.format(expected_out))
        self.__debug('Out=\n{}'.format(self.__output_data))
        diff = expected_out - self.__output_data
        # TODO(kaj): in another implementation they power up the mean -> not mean the power up
        # TODO(kaj): this shall be moved into forward propagation
        mean_squared_error = np.power(np.sum(diff, axis=1, keepdims=True), 2)
        self.__calculateCorrectionAndWeights(diff)

        old_weights = self.__weights
        # TODO(kaj): check dimension of 'correction' -> the length of it increasing alongside the samples number
        return self.__correction, self.__weights, mean_squared_error

    def propagateHiddenBackward(self, next_correction, next_weight):
        """
        Might be converted to private
        :param next_correction:
        :param next_weight:
        :return:
        """
        self.__debug('next_weight=\n{}'.format(next_weight))
        self.__debug('next_correction=\n{}'.format(next_correction))
        difference_increase = next_correction.dot(next_weight)
        self.__calculateCorrectionAndWeights(difference_increase)

        return self.__correction, self.__weights

    def output(self) -> np.array:
        """
        Might be converted to private
        :return:
        """
        return self.__output_data

    def __calculateCorrectionAndWeights(self, difference_increase):
        self.__debug('difference_increase=\n{}'.format(difference_increase))
        self.__correction = self.__activation_function.differentiate(difference_increase, self.__output_data)

        self.__debug('Learning ratio=\n{}'.format(self.__learning_ratio))
        self.__debug('Correction=\n{}'.format(self.__correction))
        self.__debug('Input=\n{}'.format(self.__input_data))
        self.__delta_weights = self.__learning_ratio * self.__correction.T.dot(self.__input_data)

        self.__debug('Delta weights=\n{}'.format(self.__delta_weights))
        self.__weights = self.__weights + self.__delta_weights

        self.__debug('New Weights=\n{}'.format(self.__weights))

    def __debug(self, msg, *args):
        if self.__debug_on:
            print('{:>12s}: '.format(self.__name), msg, *args)
