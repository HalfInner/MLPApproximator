#  Copyright (c) 2020
#  Kajetan Brzuszczak
import numpy as np

from MLPApproximator.MlpFunctionGenerator import TestingSet
from MLPApproximator.MlpMetrics import MLPMetrics
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

        self.__bias_number = 1
        self.__input_number = input_number
        self.__output_number = output_number

        self.__debug(' input number={}'.format(input_number))
        self.__debug('output number={}'.format(output_number))
        self.__debug('hidden number={}'.format(hidden_layer_number))

        self.__p1 = Perceptron(
            input_number + self.__bias_number,
            hidden_layer_number + self.__bias_number,
            activation_function=activation_function_hidden_layer,
            debug_on=debug_on,
            weight=hidden_layer_weights,
            name="P_hide")
        self.__p2 = Perceptron(
            hidden_layer_number + self.__bias_number,
            output_number,
            activation_function=activation_function_output_layer,
            debug_on=debug_on,
            weight=output_layer_weights,
            name="P_out")
        self.__mean_output_error = None
        self.__output = None

    def train(self, train_data_set: TestingSet, epoch_number=1):
        if epoch_number <= 0:
            raise RuntimeError('Epoch must be at least one')

        if np.any(np.abs(np.array(np.array([train_data_set.X, train_data_set.Y])) > 1.)):
            raise RuntimeError('Expected dataset must be in range [-1; 1]')

        inputs_with_bias = self.__create_input_with_biases(train_data_set)

        metrics = MLPMetrics()
        for epoch in range(epoch_number):
            self.__debug('################################################')
            self.__debug('Current epoch: ', epoch)
            self.__debug('################################################')
            self.__debug('Forward Propagation')
            self.__output = self.propagateForward(inputs_with_bias)

            mean_squared_error = self.__calculate_rmse(train_data_set.Output)

            self.__debug('Backward Error Propagation')
            correction = self.propagateErrorBackward(train_data_set.Output)

            metrics.addCorrection(correction)
            metrics.addMeanSquaredError(mean_squared_error)

        self.__debug('Current denormalized output ', self.__p2.output())
        # self.__output = self.__denormalize_output(self.__p2.output())
        self.__output = self.__p2.output()
        return self.__output, metrics

    def __create_input_with_biases(self, train_data_set):
        inputs_with_bias = train_data_set.Input
        if self.__bias_number > 0:
            inputs_with_bias = np.append(
                train_data_set.Input,
                np.ones(shape=(train_data_set.Input.shape[0], self.__bias_number)),
                axis=1)
        return inputs_with_bias

    def test(self, test_data_set: TestingSet):
        self.__debug('################################################')
        self.__debug('Testing: ')
        inputs_with_bias = self.__create_input_with_biases(test_data_set)

        self.__output = self.propagateForward(inputs_with_bias)
        loss = self.__calculate_rmse(test_data_set.Output)
        return self.__output, loss

    def __calculate_rmse(self, expected_out):
        return np.sqrt(np.mean(0.5 * np.square(expected_out - self.__p2.output()), axis=0, keepdims=True)).T

    def propagateForward(self, input_data):
        """
        Might be converted to private

        :param input_data:
        """
        p1_out = self.__p1.forwardPropagation(input_data)
        return self.__p2.forwardPropagation(p1_out)

    def propagateErrorBackward(self, expected_output_data: np.array):
        """
        Might be converted to private

        :param expected_output_data:
        """
        correction, weight = self.__p2.propagateBackward(expected_output_data)
        self.__p1.propagateHiddenBackward(correction, weight)

        return correction

    def __debug(self, msg, *args):
        if self.__debug_on:
            print('Approximator: ', msg, *args)

    def __normalize_data_input(self, data_set: np.array):
        extending_ratio = 1.1
        self.__min_in = np.min(data_set - extending_ratio)
        self.__max_in = np.max(data_set + extending_ratio)

        self.__debug('IN  Min={}\tMax={}'.format(self.__min_in, self.__max_in))
        return (data_set - self.__min_in) / (self.__max_in - self.__min_in)

    def __normalize_data_output(self, data_set: np.array):
        extending_ratio = 1.1
        self.__min_out = np.min(data_set - extending_ratio)
        self.__max_out = np.max(data_set + extending_ratio)

        self.__debug('OUT Min={}\tMax={}'.format(self.__min_out, self.__max_out))
        return (data_set - self.__min_out) / (self.__max_out - self.__min_out)

    def __denormalize_output(self, data_set):
        return data_set * (self.__max_out - self.__min_out) + self.__min_out
