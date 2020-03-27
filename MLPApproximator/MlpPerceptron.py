#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np

from MLPApproximator.MlpActivationFunction import SigmoidActivationFunction


class Perceptron:
    """
    Perceptron must be controlled by upper layer
    """

    def __init__(self, input_number, output_number, activation_function=SigmoidActivationFunction(),
                 debug_on=False) -> None:
        self.__debug_on = debug_on
        self.__input_number = input_number
        self.__output_number = output_number
        self.__activation_function = activation_function

        self.__weights = np.ones((output_number, input_number), dtype=float)

        self.__output_data = np.zeros_like(self.__weights)
        self.__mean_squared_error = None
        self.__input_data = None

    def train(self):
        """
        Train the perceptron.
        """
        if self.__input_data is None:
            raise RuntimeError('Cannot proceed train without input')
        delta = 0.1 * self.__mean_squared_error @ self.__input_data.transpose()
        self.__weights += delta
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
        raw_output = self.__weights @ self.__input_data
        self.__output_data = self.__activation_function(raw_output)

        return self

    def meanSquaredErrorOutput(self, expected_out):
        """
        Might be converted to private

        :param expected_out:
        :return:
        """
        if not expected_out.shape == self.__output_data.shape:
            raise ValueError("Shape of validator must be same as the output")
        step1 = expected_out - self.__output_data
        step2 = np.ones([self.__output_number, 1]) - self.__output_data
        step3 = self.__output_data

        self.__mean_squared_error = step1 * step2 * step3
        return self.__mean_squared_error

    def meanSquaredErrorHidden(self, next_weight, next_mean_squared_error):
        """
        Might be converted to private
        :param next_weight:
        :param next_mean_squared_error:
        :return:
        """
        self.__debug('Next Weight \n', next_weight)
        self.__debug('next_mean_squared_error \n', next_mean_squared_error)
        # sys.exit(-1)
        step1 = next_weight.transpose() @ next_mean_squared_error
        step2 = np.ones([self.__output_number, 1]) - self.__output_data
        step3 = self.__output_data

        self.__mean_squared_error = step1 * step2 * step3
        self.__debug('Sq : \n', self.__mean_squared_error)
        return self.__mean_squared_error

    def output(self) -> np.array:
        """
        Might be converted to private
        :return:
        """
        return self.__output_data

    def __debug(self, msg, *args):
        if self.__debug_on:
            print(msg, *args)
