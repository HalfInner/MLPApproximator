#  Copyright (c) 2020
#  Kajetan Brzuszczak
import argparse

import numpy as np

from MLPApproximator.MlpApproximator import MlpApproximator


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
                            const=True,
                            help='Activate normalization over data set into range [0,1]. Default True.')

        parser.add_argument('-n', '--hidden_layer_neurons', dest='HiddenLayerNeurons', action='store_const',
                            const=3,
                            help='Number of neurons on hidden layer. Default 3.')

        parser.add_argument('-b', '--use_biases', dest='Biases', action='store_const',
                            const=True,
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

        parser.add_argument('-ds', '--data_set', dest='DataSetFile', action='store',
                            type=argparse.FileType('r'),
                            help='First line is a header with metadata. Determine number of input and number of output'
                                 '\'4 4\' in first line. Means that there is 4 inputs and 4 outputs'
                                 'Parser accepts text file where each column is separated by whitespace(' '). '
                                 'Each row is separated by new line(\'\\n\'). '
                                 'One column is interpreted as one input or one output.')

        parser.add_argument('-arg_f1', '--arguments_function_1', dest='f_1', action='append_const',
                            const=sum, default=False,
                            help='Generate function. Polynomials Representation. Each number represent one of factors. '
                                 'Counting from right to left. To avoid factor, use 0')
        parser.add_argument('-arg_f2', '--arguments_function_2', dest='f_2', action='append_const',
                            const=sum, default=False,
                            help='Generate function. Polynomials Representation. Each number represent one of factor. '
                                 'Counting from right to left. To avoid factor, use 0')
        parser.add_argument('-arg_f3', '--arguments_function_3', dest='f_3', action='append_const',
                            const=sum, default=False,
                            help='Generate function. Polynomials Representation. Each number represent one of factor. '
                                 'Counting from right to left. To avoid factor, use 0')

        parser.add_argument('-l1', '--log_level_1', dest='LogLevel1On', action='store_const',
                            const=sum, default=False,
                            help='Activate Simple Logging During Test. Default True')

        parser.add_argument('-l2', '--log_level_2', dest='LogLevel2On', action='store_const',
                            const=sum, default=False,
                            help='Activate Extended Logging During Test. This option includes matricies, results of '
                                 'forward propagation alongside with backward propagation. Default False')

        parser.add_argument('--version', action='version', version='%(prog)s 0.1a')

        self.__parser = parser

    def run(self, argv) -> int:
        self.__parser.parse_args(argv)
        print(self.__parser)
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
