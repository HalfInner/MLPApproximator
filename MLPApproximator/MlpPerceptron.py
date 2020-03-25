import numpy as np

from MLPApproximator.MlpActivationFunction import SigmoidActivationFunction


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