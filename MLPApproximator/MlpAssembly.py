#  Copyright (c) 2020
#  Kajetan Brzuszczak
import argparse
from fractions import Fraction

from MLPApproximator.MlpActivationFunction import TanhActivationFunction, SigmoidActivationFunction, \
    ReLUActivationFunction, LinearActivationFunction
from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import FunctionGenerator, TestingSet
from MLPApproximator.MlpUtils import MlpUtils


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
    EXIT_OK = 0

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='MLP Neural Network Function Approximation.\n'
                        'E.g. usage:\n\tpython %(prog)s -arg_f1 1/2 -arg_f1 2 -arg_f1 0 -arg_f2 3 -arg_f1 2 -arg_f3 1')

        parser.add_argument('-n', '--hidden_layer_neurons', dest='HiddenLayerNeurons', action='store',
                            default=3, type=int,
                            help='Number of neurons on hidden layer. Default 3.')

        parser.add_argument('-b', '--use_biases', dest='UseBiases', action='store',
                            default=True, type=self.__str2bool, nargs='?',
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-e', '--epoch_number', dest='EpochNumber', action='store',
                            default=3, type=int,
                            help='Set number of training iterations. Default 3.')

        parser.add_argument('-s', '--sample_number', dest='SampleNumber', action='store',
                            default=100, type=int,
                            help='Set number of samples to generate by -arg_f[1,2,3]. Default 3.')

        parser.add_argument('-r', '--ratio', dest='Ratio', action='store',
                            default=5, type=int,
                            help='Set ratio of splitting dataset. Threat each r sample sa test set. Default 5 (1:4).')

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

        parser.add_argument('-norm', '--normalize_set', dest='NormalizeSet', action='store',
                            default=True, type=self.__str2bool, nargs='?',
                            help='Activate normalization over data set into range [0,1]. '
                                 'Data set must be provided first Default True.')

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

        parser.add_argument('-print_raw', dest='PrintRawOn', action='store',
                            default=False, type=self.__str2bool, nargs='?',
                            help='Print Raw Results to console. Default False')

        parser.add_argument('-plot', dest='PlotOn', action='store',
                            default=True, type=self.__str2bool, nargs='?',
                            help='Generates learning charts after work. Default True')

        parser.add_argument('-plot_to_file', dest='PlotToFile', action='store', default=None,
                            type=argparse.FileType('r'),
                            help='Generates learning charts after work to destination'
                                 '\'-plot\' must be set to True. Default None')

        parser.add_argument('--version', action='version', version='%(prog)s 0.1a')

        self.__parser = parser
        self.__training_functions = []
        self.__input_number = 0

    def run(self, argv) -> int:
        args, unknown = self.__parser.parse_known_args(argv)

        to_file = False
        file_name = None
        if args.PlotToFile is not None:
            to_file = True
            file_name = args.PlotToFile

        self.__add_function_to_generator(args.f_1)
        self.__add_function_to_generator(args.f_2)
        self.__add_function_to_generator(args.f_3)

        training_function_generator = FunctionGenerator()
        for function in self.__training_functions:
            training_function_generator.addFunction(function)

        required_samples = args.SampleNumber
        training_set = training_function_generator.generate(required_samples)

        ratio = args.Ratio
        input_number = output_number = self.__input_number
        mlp_utils = MlpUtils()
        fitting_set_x, fitting_set_y, testing_set_x, testing_set_y = mlp_utils.split_data_set(
            input_number, output_number, ratio, required_samples, training_set)

        activation_function_map = {
            'tanh': TanhActivationFunction(),
            'sigmoid': SigmoidActivationFunction(),
            'relu': ReLUActivationFunction(),
            'linear': LinearActivationFunction(),
            None: SigmoidActivationFunction()
        }

        hidden_layer_activation_function = activation_function_map[args.HiddenLayerFunction]
        output_layer_activation_function = activation_function_map[args.OutputLayerFunction]

        hidden_layer_number = args.HiddenLayerNeurons
        epoch_number = args.EpochNumber
        mlp_approximator = MlpApproximatorBuilder() \
            .setInputNumber(input_number) \
            .setHiddenLayerNumber(hidden_layer_number) \
            .setOutputNumber(output_number) \
            .setActivationFunctionForHiddenLayer(hidden_layer_activation_function) \
            .setActivationFunctionForOutputLayer(output_layer_activation_function) \
            .setDebugMode(args.LogLevel1On) \
            .setVerboseDebugMode(args.LogLevel2On) \
            .setUseBiases(args.UseBiases) \
            .build()

        learned_outputs, metrics = mlp_approximator.train(
            TestingSet([fitting_set_x, fitting_set_y]),
            epoch_number=epoch_number)

        tested_output, loss = mlp_approximator.test(TestingSet([testing_set_x, testing_set_y]))

        plot_name = '{:>3}: M={} Hidden={} Epochs={}'.format(
            1, 3, hidden_layer_number, epoch_number)

        if args.PlotOn:
            mlp_utils.plot_rmse(epoch_number, file_name, metrics, plot_name, to_file)
            mlp_utils.plot_learning_approximation(
                file_name, fitting_set_x, fitting_set_y, learned_outputs, metrics, plot_name, to_file)
            mlp_utils.plot_testing_approximation(
                file_name, plot_name, testing_set_x, testing_set_y, tested_output, loss, to_file)

        if args.PrintRawOn:
            print('Learned output\n', learned_outputs)
            print('Metrics\n', metrics.MeanSquaredErrors)
            print('Tested output\n', learned_outputs)
            print('Tested Loss\n', loss)

        return MlpApproximatorAssembler.EXIT_OK

    def __add_function_to_generator(self, f):
        f_is_exists = f is not None and f
        if not f_is_exists:
            return

        f_out = [float(Fraction(x)) for x in f]
        self.__training_functions.append(f_out)
        self.__input_number += 1

    def __str2bool(self, v):
        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
