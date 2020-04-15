#  Copyright (c) 2020
#  Kajetan Brzuszczak
import os
from contextlib import redirect_stdout
from datetime import date, datetime
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

        fitting_range = 100
        fitting_set_x, testing_set_x = np.array_split(training_set.X[:fitting_range].T, [fitting_range])
        fitting_set_y, testing_set_y = np.array_split(training_set.Y[:fitting_range].T, [fitting_range])

        self.__train_and_plot(fitting_set_x, fitting_set_y, parameter_m, required_samples)

    def __train_and_plot(self, fitting_set_x, fitting_set_y, parameter_m, required_samples):
        directory = self.__create_date_folder_if_not_exists()
        for sub_test_idx, group_parameter in enumerate(
                product(range(parameter_m, 10 * parameter_m, parameter_m), range(100, 1000, 100))):
            input_number = output_number = 3
            hidden_layer_number = group_parameter[0]
            epoch_number = group_parameter[1]

            file_name = '{}M{:03}_N{:03}_I{:03}_S{:04}'.format(
                directory, parameter_m, hidden_layer_number, epoch_number, required_samples)
            log_file_name = file_name + '_LOG.txt'
            with open(log_file_name, 'w') as f, redirect_stdout(f):
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

                plot_name = '{}: M={} Hidden={} Epochs={} Samples={}'.format(
                    sub_test_idx, parameter_m, hidden_layer_number, epoch_number, required_samples)
                plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[0], 'b-',
                         label='F1 RMSE')
                plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[1], 'r-',
                         label='F2 RMSE')
                plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[2], 'g-',
                         label='F3 RMSE')
                plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.AvgMeanSquaredError, 'm-',
                         label='Avg RMSE')
                plt.xlabel(plot_name)
                plt.ylim(0, np.max(metrics.MeanSquaredErrors[0]) * 1.1)
                plt.legend()
                # plt.show()
                plt.savefig('{}_MSE.png'.format(file_name))
                plt.cla()

                plt.plot(fitting_set_x.T[0], fitting_set_y.T[0], 'b-', label='F1 Expected')
                plt.plot(fitting_set_x.T[0], learned_outputs.T[0], 'y-', label='F1 Predicted')
                plt.plot(fitting_set_x.T[1], fitting_set_y.T[1], 'g-', label='F2 Expected')
                plt.plot(fitting_set_x.T[1], learned_outputs.T[1], 'r-', label='F2 Predicted')
                plt.plot(fitting_set_x.T[2], fitting_set_y.T[2], 'k-', label='F3 Expected')
                plt.plot(fitting_set_x.T[2], learned_outputs.T[2], 'm-', label='F3 Predicted')
                plt.xlabel(plot_name)
                plt.ylim(-0.1, 1.1)
                plt.legend()
                # plt.show()
                plt.savefig('{}_ACC.png'.format(file_name))
                plt.cla()
            break

    def __create_date_folder_if_not_exists(self):
        today = datetime.now()
        folder_name = today.strftime('%Y%m%H%M')
        directory = '..\\TestResults\\' + folder_name + '\\'
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def test_runFromDeadline(self):
        delta = date(2020, 0o4, 0o3) - date.today()
        self.assertTrue(delta.days >= 0, 'Not this time my friend, night is long... And you\'ve missed your deadline')
