#  Copyright (c) 2020
#  Kajetan Brzuszczak

from MLPApproximator.MlpFunctionGenerator import TestingSet
from MLPApproximator.MlpPerceptron import Perceptron


class MlpApproximator:
    """
    Multi Layer Perceptron
    Approximate function shape
    """

    def __init__(self, input_number, output_number, hidden_layer_number=3, debug_on=False):
        """

        :param input_number:
        :param output_number:
        :param hidden_layer_number:
        :param debug_file:
        """
        self.__debug_on = debug_on
        self.__input_number = input_number
        self.__output_number = output_number
        self.__p1 = Perceptron(input_number, hidden_layer_number, debug_on=debug_on)
        self.__p2 = Perceptron(hidden_layer_number, output_number, debug_on=debug_on)
        self.__mean_output_error = None

    def train(self, train_data_set: TestingSet, epoch_number=1):
        if epoch_number <= 0:
            raise RuntimeError('Epoch must be at least one')

        for epoch in range(epoch_number):
            self.propagateForward(train_data_set.Input)
            self.__mean_output_error = self.__p2.meanSquaredErrorOutput(train_data_set.Output)

            self.__p2.train()
            self.__p1.meanSquaredErrorHidden(self.__p2.weights(), self.__mean_output_error)
            self.__p1.train()

    def test(self, test_data_set):  # TODO (kaj): Begin with training the neural network
        pass

    def meanSquaredError(self):
        return self.__mean_output_error

    def propagateForward(self, input_data):
        """
        Might be converted to private

        :param input_data:
        """
        self.__p2.forwardPropagation(self.__p1.forwardPropagation(input_data).output())

    def propagateErrorBackward(self, expected_output_data):
        """
        Might be converted to private

        :param expected_output_data:
        """
        self.__p2.train()
        self.__p1.meanSquaredErrorHidden(self.__p2.weights(), p2_error)
        self.__p1.train()

    def __debug(self, msg, *args):
        if self.__debug_on:
            print(msg, *args)
