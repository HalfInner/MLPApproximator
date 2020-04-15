#  Copyright (c) 2020
#  Kajetan Brzuszczak
from timeit import default_timer as timer

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
                 activation_function_hidden_layer, activation_function_output_layer,
                 debug_level_1_on=False, debug_level_2_on=False,
                 hidden_layer_weights=None, output_layer_weights=None):
        """

        :param input_number:
        :param output_number:
        :param hidden_layer_number:
        :param debug_file:
        """
        self.__debug__level_1_on = debug_level_1_on
        self.__debug__level_2_on = debug_level_2_on

        self.__bias_number = 1
        self.__input_number = input_number
        self.__output_number = output_number

        self.__debug('\tMLP Function Approximator')
        self.__debug(' input number={}'.format(input_number))
        self.__debug('output number={}'.format(output_number))
        self.__debug('hidden number={}'.format(hidden_layer_number))

        self.__p1 = Perceptron(
            input_number + self.__bias_number,
            hidden_layer_number + self.__bias_number,
            activation_function=activation_function_hidden_layer,
            debug_on=debug_level_2_on,
            weight=hidden_layer_weights,
            name="P_hide")
        self.__p2 = Perceptron(
            hidden_layer_number + self.__bias_number,
            output_number,
            activation_function=activation_function_output_layer,
            debug_on=debug_level_2_on,
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

        self.__debug('Train on {} samples'.format(len(train_data_set.X)))
        metrics = MLPMetrics()
        training_time_start = timer()
        for epoch in range(epoch_number):
            epoch_time_start = timer()

            self.__debug('Epoch:{:>5}/{:<5}'.format(epoch + 1, epoch_number))
            self.__output = self.propagateForward(inputs_with_bias)
            loss = self.__calculate_rmse(train_data_set.Output)
            correction = self.propagateErrorBackward(train_data_set.Output)

            epoch_time_stop = timer()

            metrics.addCorrection(correction)
            metrics.addMeanSquaredError(loss)
            self.__debug('\tEpoch Time={:2.3}s GlobalTime={:2.3}s Loss={:2.3}%\n'.format(
                epoch_time_stop - epoch_time_start,
                epoch_time_stop - training_time_start,
                np.mean(loss * 100)))

        self.__debug('\tTraining Time={:2.3}s\n'.format(timer() - training_time_start))
        if self.__debug__level_2_on:
            self.__debug('\tCurrent denormalized output=\n{}\n'.format(self.__p2.output()))
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
        self.__debug('Testing: ')
        inputs_with_bias = self.__create_input_with_biases(test_data_set)

        self.__output = self.propagateForward(inputs_with_bias)
        loss = self.__calculate_rmse(test_data_set.Output)
        self.__debug('\tLoss={:2.3}%'.format(np.mean(loss) * 100))
        if self.__debug__level_2_on:
            self.__debug('\tCurrent denormalized output=\n{}'.format(self.__p2.output()))
        return self.__output, loss

    def __calculate_rmse(self, expected_out):
        loss = np.sqrt(np.mean(0.5 * np.square(expected_out - self.__p2.output()), axis=0, keepdims=True)).T
        if self.__debug__level_2_on:
            self.__debug('Loss=\n{}'.format(loss))
        return loss

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
        if self.__debug__level_1_on:
            print('Approximator: ', msg, *args)
