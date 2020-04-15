#  Copyright (c) 2020
#  Kajetan Brzuszczak
from datetime import date
from itertools import product
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from MLPApproximator.MlpActivationFunction import TanhActivationFunction, SigmoidActivationFunction
from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import FunctionGenerator, TestingSet


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

        # TODO(kaj): provide random function generator (not necessary?)
        training_functions = [
            [1, 1, 0],
            [1, -1, 0],
            [-2, 2, 0]
        ]

        self.assertEqual(parameter_m, len(training_functions[0]), "This tests requires M-parameter functions")

        training_function_generator = FunctionGenerator()
        for function in training_functions:
            training_function_generator.addFunction(function)

        required_samples = 130
        training_set = training_function_generator.generate(required_samples)

        # normalizing
        fitting_range = 100
        test_range = 20
        fitting_set_x = training_set.X[:fitting_range].T
        fitting_set_y = training_set.Y[:fitting_range].T

        # samples = max_samples
        # x = np.arange(samples).reshape([samples, 1]) * 2 * np.pi / samples - np.pi
        # inputs = np.ascontiguousarray(x, dtype=float)
        # # f_x = lambda val: np.sin(val) + 2.
        # f_x = lambda x_in: (1 / 20) * (x_in + 4) * (x_in + 2) * (x_in + 1) * (x_in - 1) * (x_in - 3) + 2
        # outputs = f_x(inputs)

        results = []

        for sub_test_idx, group_parameter in enumerate(
                product(range(parameter_m, 10 * parameter_m), range(100, 1000, 10))):
            input_number = output_number = 3
            hidden_layer_number = group_parameter[0]
            # epoch_number = group_parameter[1]
            epoch_number = 1

            mlp_approximator = MlpApproximatorBuilder() \
                .setInputNumber(input_number) \
                .setHiddenLayerNumber(hidden_layer_number) \
                .setOutputNumber(output_number) \
                .setActivationFunctionForHiddenLayer(TanhActivationFunction()) \
                .setActivationFunctionForOutputLayer(SigmoidActivationFunction()) \
                .setDebugMode(True) \
                .build()

            learned_outputs, metrics = mlp_approximator.train(
                TestingSet([fitting_set_x, fitting_set_y]),
                epoch_number=epoch_number)

            plot_name = '{}: Epochs={} Samples={} HiddenNeurons={}'.format(
                sub_test_idx, epoch_number, required_samples, hidden_layer_number)
            plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[0], '-',
                     label='Rooted Mean Squared Error')
            plt.xlabel(plot_name)
            plt.ylim(0, np.max(metrics.MeanSquaredErrors[0]) * 1.1)
            plt.legend()
            plt.show()
            # path = 'C:\\Users\\kajbr\\OneDrive\\Dokumenty\\StudyTmp\\'
            # plt.savefig('{}{:03}MSE.png'.format(path, max_samples))
            # plt.cla()

            # plt.plot(fitting_set_x.T[0], fitting_set_y.T[0], 'b.', label='1 Expected')
            # plt.plot(fitting_set_x.T[0], learned_outputs.T[0], 'r.', label='1 Predicted')
            # plt.plot(fitting_set_x.T[1], fitting_set_y.T[1], 'b.', label='2 Expected')
            # plt.plot(fitting_set_x.T[1], learned_outputs.T[1], 'r.', label='2 Predicted')
            # plt.plot(fitting_set_x.T[2], fitting_set_y.T[2], 'b.', label='3 Expected')
            # plt.plot(fitting_set_x.T[2], learned_outputs.T[2], 'r.', label='3 Predicted')
            plt.xlabel(plot_name)
            plt.ylim(-10., 10)
            plt.legend()
            plt.show()
            # plt.savefig('{}{:03}ACC.png'.format(path, max_samples))
            # plt.cla()

            break

    def test_runFromDeadline(self):
        delta = date(2020, 0o4, 0o3) - date.today()
        self.assertTrue(delta.days >= 0, 'Not this time my friend, night is long... And you\'ve missed your deadline')
