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

        normalized_train_data_input = self.__normalize_data_input(train_data_set.Input)
        # TODO(kaj): Preceptron's responsibility
        normalized_train_data_input = np.append(
            normalized_train_data_input, -np.ones(shape=(normalized_train_data_input.shape[0], 1)), axis=1)

        normalized_train_data_output = self.__normalize_data_output(train_data_set.Output)

        metrics = MLPMetrics()
        for epoch in range(epoch_number):
            self.__debug('Current epoch: ', epoch)

            epoch_output = np.array([[]])
            epoch_mse = np.array([[]])
            for normalized_input, normalized_output in zip(normalized_train_data_input, normalized_train_data_output):
                self.__debug('Forward Propagation')
                cur_output = self.propagateForward(normalized_input.reshape(1, -1))
                epoch_output = np.append(epoch_output, cur_output, axis=1)

                self.__debug('Backward Error Propagation')
                correction, mean_squared_error = self.propagateErrorBackward(normalized_output)
                epoch_mse = np.append(epoch_mse, mean_squared_error.reshape(1, -1))

            # metrics.addCorrection(correction)
            metrics.addMeanSquaredError(np.array([np.mean(epoch_mse)]))

        self.__debug('Current denormalized output ', self.__p2.output())
        self.__output = self.__denormalize_output(epoch_output)
        return self.__output, metrics

    def test(self, test_data_set):  # TODO (kaj): Begin with training the neural network
        pass

    def output(self):
        return self.__output

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
        correction, weight, mean_squared_error = self.__p2.propagateBackward(expected_output_data)
        self.__p1.propagateHiddenBackward(correction, weight)

        return correction, mean_squared_error

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
