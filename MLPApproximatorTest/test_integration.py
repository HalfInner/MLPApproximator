#  Copyright (c) 2020
#  Kajetan Brzuszczak
from contextlib import redirect_stdout
from itertools import product
from unittest import TestCase

from MLPApproximator.MlpActivationFunction import TanhActivationFunction, SigmoidActivationFunction
from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import FunctionGenerator, TestingSet
from MLPApproximator.MlpUtils import MlpUtils


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

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.__mlp_utils = MlpUtils()
        self.__directory = self.__mlp_utils.create_date_folder_if_not_exists()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        print('MLP Integration Approximator test has been started')

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        print('MLP Integration Approximator test has been finished')
        print('The results you will find under \'TestResults\' directory')

    def test_conductTestM3(self):
        print('Conduct M3')
        input_number = output_number = 3
        parameter_m = 3

        training_functions = [
            [1, 1, 0],  # x^2 + x
            [1, -1, 0],  # x^2 - x
            [-2, 2, 0]  # -2x^2 + 2x
        ]

        self.assertEqual(parameter_m, len(training_functions[0]),
                         'This tests requires {}-parameter functions'.format(parameter_m))

        training_function_generator = FunctionGenerator()
        for function in training_functions:
            training_function_generator.addFunction(function)

        print('Used functions: \n', training_function_generator.to_string())
        required_samples = 130
        training_set = training_function_generator.generate(required_samples)

        ratio = 5
        fitting_set_x, fitting_set_y, testing_set_x, testing_set_y = self.__mlp_utils.split_data_set(
            input_number, output_number, ratio, required_samples, training_set)

        self.__train_and_plot(fitting_set_x, fitting_set_y, testing_set_x, testing_set_y, parameter_m)

    def test_conductTestM5(self):
        print('Conduct M5')
        input_number = output_number = 3
        parameter_m = 5

        training_functions = [
            [-7, 21, 0, -8, 0.1],  # -7x4 + 21x3 – 8x + 24
            [-20, 0, 0, 0, 2],  # -20x^4 + 2
            [3, 0, -4, 0, -9],  # 3x^4 - 4x^2 - 9
        ]

        self.assertEqual(parameter_m, len(training_functions[0]),
                         'This tests requires {}-parameter functions'.format(parameter_m))

        training_function_generator = FunctionGenerator()
        for function in training_functions:
            training_function_generator.addFunction(function)

        print('Used functions: \n', training_function_generator.to_string())
        required_samples = 130
        training_set = training_function_generator.generate(required_samples)

        ratio = 5
        fitting_set_x, fitting_set_y, testing_set_x, testing_set_y = self.__mlp_utils.split_data_set(
            input_number, output_number, ratio, required_samples, training_set)

        self.__train_and_plot(fitting_set_x, fitting_set_y, testing_set_x, testing_set_y, parameter_m)

    def test_conductTestM7(self):
        print('Conduct M7')
        input_number = output_number = 3
        parameter_m = 7

        training_functions = [
            [-3, 0, -7, 21, 0, -8, 0.4],  # -3x6 – 7x4 + 21x3 – 8x + 0.4
            [1, 0, 0, -10, 0, 0, 0.2],  # x^6 -x10x^2 + 0.2
            [-1, 1, 0, 0, 4, 0, -10],  # -2x^6 + x^5 + 4x^2 - 10
        ]
        self.assertEqual(parameter_m, len(training_functions[0]),
                         'This tests requires {}-parameter functions'.format(parameter_m))

        training_function_generator = FunctionGenerator()
        for function in training_functions:
            training_function_generator.addFunction(function)

        print('Used functions: \n', training_function_generator.to_string())
        required_samples = 130
        training_set = training_function_generator.generate(required_samples)

        ratio = 5
        fitting_set_x, fitting_set_y, testing_set_x, testing_set_y = self.__mlp_utils.split_data_set(
            input_number, output_number, ratio, required_samples, training_set)

        self.__train_and_plot(fitting_set_x, fitting_set_y, testing_set_x, testing_set_y, parameter_m)

    def __train_and_plot(self, fitting_set_x, fitting_set_y, testing_set_x, testing_set_y, parameter_m):
        directory = self.__directory
        for sub_test_idx, group_parameter in enumerate(
                product(range(parameter_m, 10 * parameter_m + 1, parameter_m), range(100, 1000 + 1, 100))):
            with self.subTest(i=sub_test_idx):
                input_number = output_number = 3
                hidden_layer_number = group_parameter[0]
                epoch_number = group_parameter[1]
                required_samples = fitting_set_x.shape[0] + testing_set_x.shape[0]

                print('Current MLP: {:>3}. M={} N={:2} I={:>4} S={:3}'.format(
                    sub_test_idx, parameter_m, hidden_layer_number, epoch_number, required_samples))

                file_name = '{}M{:}_N{:03}_I{:03}_S{:04}'.format(
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
                        .setVerboseDebugMode(False) \
                        .build()

                    learned_outputs, metrics = mlp_approximator.train(
                        TestingSet([fitting_set_x, fitting_set_y]),
                        epoch_number=epoch_number)

                    to_file = True

                    plot_name = '{:>3}: M={} Hidden={} Epochs={}'.format(
                        sub_test_idx, parameter_m, hidden_layer_number, epoch_number)
                    self.__mlp_utils.plot_rmse(epoch_number, file_name, metrics, plot_name, to_file)

                    self.__mlp_utils.plot_learning_approximation(
                        file_name, fitting_set_x, fitting_set_y, learned_outputs, metrics, plot_name, to_file)

                    test_output, loss = mlp_approximator.test(TestingSet([testing_set_x, testing_set_y]))
                    self.__mlp_utils.plot_testing_approximation(
                        file_name, plot_name, testing_set_x, testing_set_y, test_output, loss, to_file)
