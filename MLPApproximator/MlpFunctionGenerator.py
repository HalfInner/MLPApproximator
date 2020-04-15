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

    def __init__(self, data_set) -> None:
        self.__data_set = copy(data_set)

    @property
    def X(self):
        return self.__data_set[0]

    @property
    def Y(self):
        return self.__data_set[1]

    @property
    def Input(self):
        return self.__data_set[0]

    @property
    def Output(self):
        return self.__data_set[1]


class FunctionGenerator:

    def __init__(self) -> None:
        self.__function_store = []
        pass

    def addFunction(self, polynomial):
        """

        :param polynomial: group of consecutive polynomial
            e.g. [2, 3, 1] => 2x^2 + 3x^1 + 1
        """
        self.__function_store.append(polynomial)

    def generate(self, samples_number=1):
        input_set = np.linspace(-1., 1., samples_number, dtype=float)

        output_set = np.empty((0, samples_number), dtype=float)
        for idx, polynomial in enumerate(self.__function_store):
            f_x = lambda x: sum([factor * x ** step for step, factor in enumerate(reversed(polynomial))])

            output_set_row = f_x(input_set)
            if len(output_set_row) > 1:
                output_set_row = (output_set_row - np.min(output_set_row)) / np.ptp(output_set_row)
            else:
                output_set_row = np.array([0.5])
            output_set = np.append(output_set, output_set_row.reshape(1, samples_number), axis=0)
        input_set.resize(len(self.__function_store), samples_number)
        return TestingSet([input_set, output_set])

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
