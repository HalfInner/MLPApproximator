#  Copyright (c) 2020
#  Kajetan Brzuszczak

from MLPApproximator.MlpPerceptron import Perceptron


class MlpApproximator:
    """
    Multi Layer Perceptron
    Approximate function shape
    """

    def __init__(self, input_number, output_number, hidden_layer_number=3):
        self.__input_number = input_number
        self.__output_number = output_number
        self.__p1 = Perceptron(input_number, hidden_layer_number)
        self.__p2 = Perceptron(hidden_layer_number, output_number, test_first_p1=False)
        self.__mean_squared_error = None

    def propagateForward(self, input_data):
        """

        :param input_data:
        """
        self.__p2.forwardPropagation(self.__p1.forwardPropagation(input_data).output())

    def propagateErrorBackward(self, expected_output_data):
        """

        :param expected_output_data:
        """
        p2_error = self.__p2.meanSquaredErrorOutput(expected_output_data)
        self.__p2.train(expected_output_data)
        # p2.weights * output.p2
        # 1 - output.p1
        #  output.p1
        self.__p1.meanSquaredErrorHidden(self.__p2.weights(), p2_error)
        self.__p1.train()
