#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np

from MLPApproximator.MlpApproximator import MlpApproximator


class TestingSet:

    def __init__(self, set) -> None:
        self.__set = set

    @property
    def X(self):
        return self.__set[0]

    @property
    def Y(self):
        return self.__set[1]

    @property
    def Input(self):
        return self.__set[0]

    @property
    def Output(self):
        return self.__set[1]


class FunctionGenerator:

    def __init__(self) -> None:
        self.__function_store = []
        pass

    def addFunction(self, polynomial):
        """

        :param polynomial: group of consecutive polynomial
            e.g. [2, 3, 1] > 2x^2 + 3x^1 + 1
        """
        self.__function_store.append(polynomial)

    def generate(self, samples_number=1):
        input_set = np.zeros([samples_number, len(self.__function_store)], dtype=float)
        output_set = np.zeros([samples_number, len(self.__function_store)], dtype=float)

        for x in range(samples_number):
            for idx, polynomial in enumerate(self.__function_store):
                value = sum([factor * x ** step for step, factor in enumerate(reversed(polynomial))])
                input_set[idx][x] = x
                output_set[idx][x] = value

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


class MlpApproximatorAssembler:
    """
    Assembly modules from the project

    example::
        def run(self) -> str:
            first_sample = np.array([1, 2]).reshape((2, 1))
            expected_out = np.array([1, 0]).reshape((2, 1))

            mlp_approximator = MlpApproximator(2, 2, 3)
            mlp_approximator.propagateForward(first_sample)
            mlp_approximator.propagateErrorBackward(expected_out)

            return 0
         ``
    """

    def run(self) -> str:
        """
        :return: error code 0-OK
        """
        # sample: x=(1,2); output: f(x) = (1, 0).
        first_sample = np.array([1, 2]).reshape((2, 1))
        expected_out = np.array([1, 0]).reshape((2, 1))

        mlp_approximator = MlpApproximator(2, 2, 3)
        mlp_approximator.propagateForward(first_sample)
        mlp_approximator.propagateErrorBackward(expected_out)

        return 0
