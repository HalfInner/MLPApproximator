#  Copyright (c) 2020
#  Kajetan Brzuszczak

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
        self.__mean_squared_error = None
        self.__input_data = None

    def train(self):
        if self.__input_data is None:
            raise RuntimeError('Cannot proceed train without input')
        delta = 0.1 * self.__mean_squared_error @ self.__input_data.transpose()
        self.__weights += delta
        print('Weights \n', self.__weights)

    def weights(self):
        return self.__weights

    def forwardPropagation(self, input_data):
        self.__input_data = input_data
        raw_output = self.__weights @ self.__input_data
        self.__output_data = self.__activation_function(raw_output)

        return self

    def meanSquaredErrorOutput(self, expected_out):
        if not expected_out.shape == self.__output_data.shape:
            raise ValueError("Shape of validator must be same as the output")
        step1 = expected_out - self.__output_data
        step2 = np.ones([self.__output_number, 1]) - self.__output_data
        step3 = self.__output_data

        self.__mean_squared_error = step1 * step2 * step3
        return self.__mean_squared_error

    def meanSquaredErrorHidden(self, next_weight, next_mean_squared_error):
        print('Next Weight \n', next_weight)
        print('next_mean_squared_error \n', next_mean_squared_error)
        # sys.exit(-1)
        step1 = next_weight.transpose() @ next_mean_squared_error
        step2 = np.ones([self.__output_number, 1]) - self.__output_data
        step3 = self.__output_data

        self.__mean_squared_error = step1 * step2 * step3
        print('Sq : \n', self.__mean_squared_error)
        return self.__mean_squared_error


    def backwardPropagation(self, validate_out):
        pass

    def output(self) -> np.array:
        return self.__output_data
