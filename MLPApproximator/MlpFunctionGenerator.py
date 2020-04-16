#  Copyright (c) 2020
#  Kajetan Brzuszczak
from copy import copy

import numpy as np


class TestingSet:
    """
    Wrapper for testing set.
    First matrix shall contains INPUTS/X
    Second matrix shall contains OUTPUTS/Y
    """
    X_DATA_IDX = 0
    Y_DATA_IDX = 1

    def __init__(self, data_set) -> None:
        self.__data_set = copy(data_set)

    @property
    def X(self):
        return self.__data_set[TestingSet.X_DATA_IDX]

    @property
    def Y(self):
        return self.__data_set[TestingSet.Y_DATA_IDX]

    @property
    def Input(self):
        return self.__data_set[TestingSet.X_DATA_IDX]

    @property
    def Output(self):
        return self.__data_set[TestingSet.Y_DATA_IDX]

    def to_string(self):
        parsed_data_set = "# Auto Generated Data Set by MLP Approximator \n" \
                          "# Tabs are used to seperated each data \n" \
                          "# First line contains header with the number of input and output \n" \
                          "# Remaining part is the raw data \n"
        parsed_data_set += str(len(self.__data_set[TestingSet.X_DATA_IDX][0])) + ' '
        parsed_data_set += str(len(self.__data_set[TestingSet.Y_DATA_IDX][0])) + '\n'

        for input_line, output_line in zip(self.__data_set[TestingSet.X_DATA_IDX],
                                           self.__data_set[TestingSet.Y_DATA_IDX]):
            parsed_data_set += '\t'.join(map(str, input_line)) + '\t' + '\t'.join(map(str, output_line)) + '\n'

        return parsed_data_set


class FunctionGenerator:
    def __init__(self) -> None:
        self.__function_store = []

    def addFunction(self, polynomial):
        """

        :param polynomial: group of consecutive polynomial
            e.g. [2, 3, 1] => 2x^2 + 3x^1 + 1
        """
        self.__function_store.append(polynomial)

    def generate(self, samples_number=1):
        if not self.__function_store:
            raise ValueError('You must add some functions before start')

        if samples_number <= 0:
            raise ValueError('Number of samples must be greater than 0')

        input_set = np.linspace(-1., 1., samples_number, dtype=float)
        output_set = np.empty((0, samples_number), dtype=float)
        for idx, polynomial in enumerate(self.__function_store):
            f_x = lambda x: sum([factor * x ** step for step, factor in enumerate(reversed(polynomial))])
            output_set_row = f_x(input_set)
            output_set = np.append(output_set, output_set_row.reshape(1, samples_number), axis=0)

        output_set_range = np.ptp(output_set)
        if abs(output_set_range) > 0.0000001:
            output_set = (output_set - np.min(output_set)) / output_set_range
        else:
            output_set = output_set * 0 + 0.5

        input_set = np.resize(input_set, (len(self.__function_store), samples_number))
        return TestingSet([input_set.T, output_set.T])

    def to_string(self):
        output_str = ""
        for polynomial in self.__function_store:
            step = -1
            polynomial_str = ''
            for factor in reversed(polynomial):
                step += 1
                if factor == 0:
                    continue

                local_str = ' + ' + str(factor)
                if step == 1:
                    local_str += 'x'
                if step > 1:
                    local_str += 'x^' + str(step)
                polynomial_str = local_str + polynomial_str
            output_str += polynomial_str.replace(' + ', '', 1).lstrip() + '\n'

        return output_str
