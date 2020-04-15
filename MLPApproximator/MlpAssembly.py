#  Copyright (c) 2020
#  Kajetan Brzuszczak
import argparse

from MLPApproximator.MlpActivationFunction import TanhActivationFunction, SigmoidActivationFunction
from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import FunctionGenerator


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

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='MLP Neural Network Function Approximation.\n'
                        'E.g. usage:\n\tpython %(prog)s -arg_f1 1/2 -arg_f1 2 -arg_f1 0 -arg_f2 3 -arg_f1 2 -arg_f3 1')

        parser.add_argument('-norm', '--normalize_set', dest='NormalizeSet', action='store_const',
                            const=True, default=True,
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-n', '--hidden_layer_neurons', dest='HiddenLayerNeurons', action='store_const',
                            const=3, default=3,
                            help='Number of neurons on hidden layer. Default 3.')

        parser.add_argument('-b', '--use_biases', dest='Biases', action='store_const',
                            const=True, default=True,
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-hf', '--hidden_layer_activation_function',
                            choices=['tanh', 'sigmoid', 'relu', 'linear'],
                            dest='HiddenLayerFunction',
                            type=str,
                            default='tanh',
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-of', '--output_layer_activation_function', dest='OutputLayerFunction',
                            default='sigmoid', choices=['tanh', 'sigmoid', 'relu', 'linear'],
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-ds', '--data_set', dest='DataSetFile', action='store', default=None,
                            type=argparse.FileType('r'),
                            help='First line is a header with metadata. Determine number of input and number of output'
                                 '\'2 3\' in first line. Means that there is 4 inputs and 4 outputs'
                                 'Parser accepts text file where each column is separated by whitespace(' '). '
                                 'Each row is separated by new line(\'\\n\'). '
                                 'One column is interpreted as one input or one output.'
                                 '\'1 2 3 4 5\' in second line means that first two columns colums are the input, '
                                 'and last two are the expected output')

        parser.add_argument('-arg_f1', '--arguments_function_1', dest='f_1', action='append',
                            default=[],
                            help='Generate function. Polynomials Representation. Each number represent one of factors. '
                                 'Counting from right to left. To avoid factor, use 0. Generate one output')
        parser.add_argument('-arg_f2', '--arguments_function_2', dest='f_2', action='append',
                            default=[],
                            help='Generate function. Polynomials Representation. Each number represent one of factor. '
                                 'Counting from right to left. To avoid factor, use 0. Generate one output')
        parser.add_argument('-arg_f3', '--arguments_function_3', dest='f_3', action='append',
                            default=[],
                            help='Generate function. Polynomials Representation. Each number represent one of factor. '
                                 'Counting from right to left. To avoid factor, use 0. Generate one output')

        parser.add_argument('-l1', '--log_level_1', dest='LogLevel1On', action='store_const',
                            const=sum, default=True,
                            help='Activate Simple Logging During Test. Default True')

        parser.add_argument('-l2', '--log_level_2', dest='LogLevel2On', action='store_const',
                            const=False, default=False,
                            help='Activate Verbose Logging During Test. Default False')

        parser.add_argument('-plot', dest='PlotOn', action='store_const',
                            const=True, default=True,
                            help='Generates learning charts after work. Default True')

        parser.add_argument('--version', action='version', version='%(prog)s 0.1a')

        self.__parser = parser

    def run(self, argv) -> int:
        args, unknown = self.__parser.parse_known_args(argv)

        training_functions = []
        if args.f_1 is not None:
            training_functions.append(args.f_1)

        training_function_generator = FunctionGenerator()
        for function in training_functions:
            training_function_generator.addFunction(function)

        required_samples = 130
        training_set = training_function_generator.generate(required_samples)

        ratio = 5
        fitting_set_x, fitting_set_y, testing_set_x, testing_set_y = self.__mlp_utils.split_data_set(
            input_number, ratio, required_samples, training_set)

        input_number = output_number = 1
        hidden_layer_number = 3
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

        to_file = True

        plot_name = '{:>3}: M={} Hidden={} Epochs={}'.format(
            sub_test_idx, parameter_m, hidden_layer_number, epoch_number)
        self.__mlp_utils.plot_rmse(epoch_number, file_name, metrics, plot_name, to_file)

        self.__mlp_utils.plot_learning_approximation(
            file_name, fitting_set_x, fitting_set_y, learned_outputs, metrics, plot_name, to_file)

        test_output, loss = mlp_approximator.test(TestingSet([testing_set_x, testing_set_y]))
        self.__mlp_utils.plot_testing_approximation(
            file_name, plot_name, testing_set_x, testing_set_y, test_output, loss, to_file)
