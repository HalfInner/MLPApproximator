#  Copyright (c) 2020
#  Kajetan Brzuszczak
import numpy as np

from MLPApproximator.MlpFunctionGenerator import TestingSet
from MLPApproximator.MlpPerceptron import Perceptron


class MlpApproximator:
    """
    Multi Layer Perceptron
    Approximate function shape
    """

    def __init__(self, input_number, output_number, hidden_layer_number,
                 activation_function_hidden_layer, activation_function_output_layer, debug_on=False,
                 hidden_layer_weights=None, output_layer_weights=None):
        """

        :param input_number:
        :param output_number:
        :param hidden_layer_number:
        :param debug_file:
        """
        self.__debug_on = debug_on
        self.__input_number = input_number
        self.__output_number = output_number
        self.__p1 = Perceptron(
            input_number,
            hidden_layer_number,
            activation_function=activation_function_hidden_layer,
            debug_on=debug_on,
            weight=hidden_layer_weights)
        self.__p2 = Perceptron(
            hidden_layer_number,
            output_number,
            activation_function=activation_function_output_layer,
            debug_on=debug_on,
            weight=output_layer_weights)
        self.__mean_output_error = None
        self.__output = None

    def train(self, train_data_set: TestingSet, epoch_number=1):
        if epoch_number <= 0:
            raise RuntimeError('Epoch must be at least one')

        normalized_output_data_set = self.__normalize(train_data_set.Output)
        for epoch in range(epoch_number):
            self.__debug('Current epoch: ', epoch)
            self.__output = self.propagateForward(train_data_set.Input)

            self.propagateErrorBackward(normalized_output_data_set)

        self.__debug('Current denormalized output ', self.__p2.output())
        self.__output = self.__denormalize(self.__p2.output())
        return self.__output

    def test(self, test_data_set):  # TODO (kaj): Begin with training the neural network
        pass

    def output(self):
        return self.__output

    def meanSquaredError(self):
        return self.__mean_output_error

    def propagateForward(self, input_data):
        """
        Might be converted to private

        :param input_data:
        """
        return self.__p2.forwardPropagation(self.__p1.forwardPropagation(input_data))

    def propagateErrorBackward(self, expected_output_data):
        """
        Might be converted to private

        :param expected_output_data:
        """
        # self.__p2.train()
        # self.__p1.propagateHiddenBackward(self.__p2.weights(), self.__p2.output())
        # self.__p1.train()
        next_correction, next_weight = self.__p2.propagateBackward(expected_output_data)
        self.__p1.propagateHiddenBackward(next_correction, next_weight)

    def __debug(self, msg, *args):
        if self.__debug_on:
            print('Approximator: ', msg, *args)

    def __normalize(self, data_set):
        min_from_set = np.min(data_set)
        max_from_set = np.max(data_set)
        self.__min_x = min_from_set
        self.__max_x = max_from_set
        return (data_set + min_from_set) / (max_from_set - min_from_set)

    def __denormalize(self, data_set):
        return data_set * (self.__max_x + self.__min_x) - self.__min_x
