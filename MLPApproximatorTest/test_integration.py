#  Copyright (c) 2020
#  Kajetan Brzuszczak
from datetime import date
from itertools import product
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import FunctionGenerator


class TestIntegration(TestCase):
    """
    Draft description for research:
        1) generate learning dataset - doubled - training/testing
        2) teach neural network three inputs three outputs
            3 different function with M={3,5,7} parameters
        3) for each tipple function group conduct tests - training and testing
            a) for each test change hidden layer nodes from N=M to N=10*M.
                b) for each hidden layer(?) conduct tests where you have I=100 epochs of learning heading to 1000
        4) process results for each combination
    """

    def test_conductTestM3(self):
        parameter_m = 3

        output_number = 3
        # TODO(kaj): provide random function generator
        training_functions = [
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ]

        # TODO(kaj): provide random function generator, at least in test functions
        test_functions = [
            [3, 2, 1],
            [-3, 0, 1],
            [3, -2, -1]
        ]
        self.assertEqual(parameter_m, len(training_functions[0]), "This tests requires M-parameter functions")

        training_function_generator = FunctionGenerator()

        for function in training_functions:
            training_function_generator.addFunction(function)

        test_function_generator = FunctionGenerator()

        for function in training_functions:
            test_function_generator.addFunction(function)

        required_samples = 100
        training_set = training_function_generator.generate(required_samples)
        print(training_set.Output[0])
        test_set = test_function_generator.generate(required_samples)

        results = []

        for sub_test_idx, group_parameter in enumerate(product(range(parameter_m, 10 * parameter_m), range(100, 1000))):
            parameter_n = group_parameter[0]
            parameter_i = group_parameter[1]
            print('I: ', parameter_i)
            parameter_n = 5
            print('N: ', parameter_n)

            mlp_approximator = MlpApproximatorBuilder() \
                .setInputNumber(parameter_m) \
                .setHiddenLayerNumber(parameter_n) \
                .setOutputNumber(output_number) \
                .setDebugMode(False) \
                .build()

            learn_output, metrics = mlp_approximator.train(training_set, parameter_i)

            plt.plot(metrics.MeanSquaredErrors[0], label='Mean Squared Error Out1')
            plt.plot(metrics.MeanSquaredErrors[1], label='Mean Squared Error Out2')
            plt.plot(np.mean(metrics.MeanSquaredErrors, axis=0), label='Mean Squared Error AVG')
            plt.xlabel('Epochs (Hidden Neurons={})'.format(parameter_n))
            plt.legend()
            plt.show()

            print(learn_output)
            plt.plot(training_set.Input[0], learn_output[0], label='F1')
            plt.plot(training_set.Input[1], learn_output[1], label='F2')
            plt.plot(training_set.Input[2], learn_output[2], label='F3')
            plt.xlabel('Epochs (Hidden Neurons={})'.format(parameter_n))
            plt.legend()
            plt.show()

            break


    def test_runFromDeadline(self):
        delta = date(2020, 0o4, 0o3) - date.today()
        self.assertTrue(delta.days >= 0, 'Not this time my friend, night is long... And you\'ve missed your deadline')
