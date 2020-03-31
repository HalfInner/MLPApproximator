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
        One complete iteration o learing 3-layerd peceptron (MLP). 3 neurons in hidden layer. 2 neuron on input,
        and 2 neurons on output
            * const learning ratio = 0.1
            * sigmoid activation function
            * X=(1,2)  f(x)=Y=(1,0)
            * W1=[[1 -1][1 1][-1 1]]
            * W2=[[1 -1 1][-1 1 -1]]
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
            .setDebugMode(False) \
            .setHiddenLayerWeights(w1) \
            .setOutputLayerWeights(w2) \
            .build()

        mlp_approximator.train(TestingSet([first_sample, expected_out]))
        out_epoch_1 = mlp_approximator.output()
        mlp_approximator.train(TestingSet([first_sample, expected_out]))
        out_epoch_2 = mlp_approximator.output()

        sys.stdout = original_stdout

        expected_out_1 = np.array([.51185425, .48814575]).reshape([2, 1])
        expected_out_2 = np.array([.5988137, .4011863]).reshape([2, 1])
        self.assertTrue(np.allclose(expected_out_1, out_epoch_1), 'Out1=\n{}'.format(out_epoch_1))
        self.assertTrue(np.allclose(expected_out_2, out_epoch_2), 'Out2=\n{}'.format(out_epoch_2))
