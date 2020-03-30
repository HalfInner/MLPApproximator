#  Copyright (c) 2020
#  Kajetan Brzuszczak
import sys
from io import StringIO
from unittest import TestCase

import numpy as np

from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import TestingSet


class TestMlpApproximator(TestCase):

    def test_propagateForward(self):
        """
        Not proper test, but makes sure that changes implementation won't influence on already verified basic  behaviour
        One complete iteration o learing 3-layerd peceptron (MLP). 3 neurons in hidden layer. 2 neuron on input,
        and 2 neurons on output
            * const learning ratio = 0.1
            * sigmoid activation function
            * X=(1,2)  f(x)=Y=(1,0)
            * W1=[[1 -1][1 1][-1 1]]  W2=[[1 -1 1][-1 1 -1]]
        """
        input_number = output_number = 2
        hidden_layer_number = 3

        expected_out = np.array([1, 0]).reshape((input_number, 1))
        first_sample = np.array([1, 2]).reshape((input_number, 1))

        original_stdout = sys.stdout
        sys.stdout = string_io_out = StringIO()

        w1 = np.array([1, - 1, 1, 1, -1, 1]).reshape([hidden_layer_number, input_number])
        w2 = np.array([1, - 1, 1, -1, 1, - 1]).reshape([output_number, hidden_layer_number])

        mlp_approximator = MlpApproximatorBuilder() \
            .setInputNumber(input_number) \
            .setHiddenLayerNumber(hidden_layer_number) \
            .setOutputNumber(output_number) \
            .setDebugMode(True) \
            .setHiddenLayerWeights(w1) \
            .setOutputLayerWeights(w2) \
            .build()

        mlp_approximator.train(TestingSet([first_sample, expected_out]), epoch_number=40000)
        sys.stdout = original_stdout

        # expected_out = """
        #     Weights
        #      [[1.00026546 1.00026546 1.00026546]
        #      [0.99537525 0.99537525 0.99537525]]
        #     Next Weight
        #      [[1.00026546 1.00026546 1.00026546]
        #      [0.99537525 0.99537525 0.99537525]]
        #     next_mean_squared_error
        #      [[ 0.00278674]
        #      [-0.04855007]]
        #     Sq :
        #      [[-0.00205726]
        #      [-0.00205726]
        #      [-0.00205726]]
        #     Weights
        #      [[0.99979427 0.99958855]
        #      [0.99979427 0.99958855]
        #      [0.99979427 0.99958855]]
        # """
        # self.assertEqual(expected_out.replace(' ', '').replace('\n', ''),
        #                  string_io_out.getvalue().replace(' ', '').replace('\n', ''))
        # print('Output', string_io_out.getvalue())
        print('Out: ', mlp_approximator.output())
        # self.assertEqual(np.array([1, 0]).reshape([2, 1]), mlp_approximator.output())
