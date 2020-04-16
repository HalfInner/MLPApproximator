#  Copyright (c) 2020
#  Kajetan Brzuszczak
import argparse
from fractions import Fraction
from matplotlib import pyplot as plt

import numpy as np

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
            description='MLP Neural Network Function Approximator.\n'
                        'E.g. usage:'
                        '\n\tpython %(prog)s -arg_f1 1/2 -arg_f1 2 -arg_f1 0 -arg_f2 3 -arg_f1 2 -arg_f3 1'
                        '\n\t%(prog)s -ds Examples/DataSetM5.txt -norm False -plot_to_dir Result -e 100 -n 10',
            epilog='© 2020 Kajetan Brzuszczak',
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('-b', '--use_biases', dest='UseBiases', action='store',
                            default=True, type=self.__str2bool, nargs='?', choices=[True, False],
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-e', '--epoch_number', dest='EpochNumber', action='store',
                            default=3, type=int,
                            help='Set number of training iterations. Default 3.')

        parser.add_argument('-n', '--hidden_layer_neurons', dest='HiddenLayerNeurons', action='store',
                            default=3, type=int,
                            help='Number of neurons on hidden layer. Default 3.')

        parser.add_argument('-hf', '--hidden_layer_activation_function',
                            choices=['tanh', 'sigmoid', 'relu', 'linear'],
                            dest='HiddenLayerFunction',
                            type=str,
                            default='tanh',
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-of', '--output_layer_activation_function', dest='OutputLayerFunction',
                            default='sigmoid', choices=['tanh', 'sigmoid', 'relu', 'linear'],
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-ds', '--data_set', dest='DataSetFileHandler', action='store', default=None,
                            type=argparse.FileType('r'),
                            help='First line is a header with metadata. Determine number of input and number of output'
                                 ' e.g. \'2 3\' in first line means that there is 2 inputs and 3 outputs'
                                 'Parser accepts text file where each column is separated by whitespace(\' \') '
                                 'or tabulation(\'\\t\'). Each row is separated by new line(\'\\n\'). '
                                 'One column is interpreted as one input or one output.'
                                 '\'1 2 3 4 5\' in second line means that first two columns colums are the input, '
                                 'and last two are the expected output. The \'\'#\' character threat as comment'
                                 'Using file as input excludes usage of function generator. '
                                 'Data must be normalized before otherwise it terminates itself immediately. ')

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

        parser.add_argument('-s', '--sample_number', dest='SampleNumber', action='store',
                            default=100, type=int,
                            help='Set number of samples to generate by -arg_f[1,2,3]. Default 3.')

        parser.add_argument('-r', '--ratio', dest='Ratio', action='store',
                            default=5, type=int,
                            help='Set ratio of splitting dataset. Threat each r-th sample sa test set. Default 5 (1:4)')

        parser.add_argument('-sds', '--save_data_set_to_file', dest='SaveDataSetFileHandler', action='store',
                            default=None, type=argparse.FileType('w'),
                            help='Save result of generation into the passed file. Apply to -arg_3[1,2,3]')

        parser.add_argument('-l1', '--log_level_1', dest='LogLevel1On', action='store_const',
                            const=sum, default=True,
                            help='Activate Simple Logging During Test. Default True')

        parser.add_argument('-l2', '--log_level_2', dest='LogLevel2On', action='store_const',
                            const=False, default=False,
                            help='Activate Verbose Logging During Test. Default False')

        parser.add_argument('-print_raw', dest='PrintRawOn', action='store',
                            default=False, choices=[True, False], type=self.__str2bool, nargs='?',
                            help='Print Raw Results to console. Default False')

        parser.add_argument('-plot', dest='PlotOn', action='store',
                            default=True, type=self.__str2bool, nargs='?', choices=[True, False],
                            help='Generates learning charts after work. Default True')

        parser.add_argument('-plot_to_dir', dest='PlotToDir', action='store', default=None,
                            type=str,  # TODO (kaj) : add directory validation
                            help='Generates learning charts after work to destination'
                                 '\'-plot\' must be set to True. Default None')

        parser.add_argument('--version', action='version', version='%(prog)s 0.1a')

        self.__parser = parser
        self.__training_functions = []
        self.__input_number = 0
        self.__mlp_utils = MlpUtils()

    def run(self, argv) -> int:
        args, unknown = self.__parser.parse_known_args(argv)

        save_to_file = False
        dir_name = None
        if args.PlotToDir is not None:
            save_to_file = True
            dir_name = self.__mlp_utils.create_date_folder_if_not_exists(args.PlotToDir)
        ratio = args.Ratio

        input_number, output_number, required_samples, training_set = None, None, None, None
        if not args.DataSetFileHandler:
            input_number, output_number, required_samples, training_set = self.__generate_functions(args)
        else:
            input_number, output_number, required_samples, training_set = self.__parse_data_set_file(
                args.DataSetFileHandler)

        if args.SaveDataSetFileHandler:
            args.SaveDataSetFileHandler.write(training_set.to_string())

        fitting_set_x, fitting_set_y, testing_set_x, testing_set_y = self.__mlp_utils.split_data_set(
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

        if args.PrintRawOn:
            print('Learned output\n', learned_outputs)
            print('Metrics\n', metrics.MeanSquaredErrors)
            print('Tested output\n', learned_outputs)
            print('Tested Loss\n', loss)

        if args.PlotOn:
            plot_name = '{:>3}: M={} Hidden={} Epochs={}'.format(1, 3, hidden_layer_number, epoch_number)
            self.__mlp_utils.plot_rmse(epoch_number, dir_name, metrics, plot_name, save_to_file)
            self.__mlp_utils.plot_learning_approximation(
                dir_name, fitting_set_x, fitting_set_y, learned_outputs, metrics, plot_name, save_to_file)
            self.__mlp_utils.plot_testing_approximation(
                dir_name, plot_name, testing_set_x, testing_set_y, tested_output, loss, save_to_file)

        return MlpApproximatorAssembler.EXIT_OK

    def __generate_functions(self, args):
        self.__add_function_to_generator(args.f_1)
        self.__add_function_to_generator(args.f_2)
        self.__add_function_to_generator(args.f_3)
        training_function_generator = FunctionGenerator()
        for function in self.__training_functions:
            training_function_generator.addFunction(function)
        required_samples = args.SampleNumber
        training_set = training_function_generator.generate(required_samples)
        input_number = output_number = self.__input_number
        return input_number, output_number, required_samples, training_set

    def __parse_data_set_file(self, file_handler):
        input_number, output_number = 0, 0
        required_samples = 0
        training_set = None
        is_header_read = False
        is_data_read = False
        input_data = None
        output_data = None

        for line in file_handler:
            if not line:
                continue

            elements = line.strip().replace('\t', ' ').split(' ')
            is_comment = elements[0].lstrip()[0] == '#'
            if is_comment:
                continue

            if not is_header_read:
                if len(elements) != 2:
                    raise RuntimeError('Header must contains number of inputs and outputs')
                input_number = int(elements[0])
                output_number = int(elements[1])

                input_data = np.empty((0, input_number), dtype=float)
                output_data = np.empty((0, output_number), dtype=float)

                is_header_read = True
                continue
            if len(elements) != (input_number + output_number):
                raise RuntimeError('Number of columns={} must equals to inputs number={} and outputs number={}'
                                   .format(len(elements), input_number, output_number))

            input_data = np.append(input_data, [elements[:input_number]], axis=0)
            output_data = np.append(output_data, [elements[input_number:]], axis=0)
            required_samples += 1

        testing_set = TestingSet([input_data.astype('float64'), output_data.astype('float64')])
        return input_number, output_number, required_samples, testing_set

    def __add_function_to_generator(self, f):
        f_is_exists = f is not None and f
        if not f_is_exists:
            return

        f_out = [float(Fraction(x)) for x in f]
        self.__training_functions.append(f_out)
        self.__input_number += 1

    def __str2bool(self, v):
        # TODO(kaj): common convention is use --function/--no-function instead of parsing booleans
        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
