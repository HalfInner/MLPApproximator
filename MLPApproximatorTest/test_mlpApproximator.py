#  Copyright (c) 2020
#  Kajetan Brzuszczak
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from MLPApproximator.MlpApproximatorBuilder import MlpApproximatorBuilder
from MLPApproximator.MlpFunctionGenerator import TestingSet


class TestMlpApproximator(TestCase):

    def test_propagateForward(self):
        """
        Book example: One complete iteration o learning 3-layers perceptron (MLP). 3 neurons in hidden layer. 2 neuron on input,
        and 2 neurons on output
            * const learning ratio = 0.1
            * sigmoid activation function
            * X=(1,2)  f(x)=Y=(1,0)
            * W1=[[1 -1][1 1][-1 1]]
            * W2=[[1 -1 1][-1 1 -1]]
        """
        input_number = output_number = 2
        hidden_layer_number = 3

        first_sample = np.array([1, 2]).reshape((input_number, 1))
        expected_out = np.array([1, 0]).reshape((input_number, 1))

        w1 = np.array([1, - 1, 1, 1, -1, 1]).reshape([hidden_layer_number, input_number])
        w2 = np.array([1, - 1, 1, -1, 1, -1]).reshape([output_number, hidden_layer_number])

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

        expected_out_1 = np.array([.51185425, .48814575]).reshape([2, 1])
        expected_out_2 = np.array([.5988137, .4011863]).reshape([2, 1])
        self.assertTrue(np.allclose(expected_out_1, out_epoch_1), 'Out1=\n{}'.format(out_epoch_1))
        self.assertTrue(np.allclose(expected_out_2, out_epoch_2), 'Out2=\n{}'.format(out_epoch_2))

    def test_shouldLearnScopeAimsInfinity(self):
        """Description must be"""
        input_number = output_number = 2
        hidden_layer_number = 3

        zero_number = 0.
        great_number = 10000000
        first_sample = np.array([1, 0]).reshape((input_number, 1))
        expected_out = np.array([zero_number, great_number]).reshape((input_number, 1))

        w1 = np.array([1, - 1, 1, 1, -1, 1]).reshape([hidden_layer_number, input_number])
        w2 = np.array([1, - 1, 1, -1, 1, -1]).reshape([output_number, hidden_layer_number])

        mlp_approximator = MlpApproximatorBuilder() \
            .setInputNumber(input_number) \
            .setHiddenLayerNumber(hidden_layer_number) \
            .setOutputNumber(output_number) \
            .setDebugMode(False) \
            .setHiddenLayerWeights(w1) \
            .setOutputLayerWeights(w2) \
            .build()

        mlp_approximator.train(TestingSet([first_sample, expected_out]), 100)
        out_epoch_1 = mlp_approximator.output()
        expected_out_1 = np.array([609392.02633787, 9390607.97366213]).reshape([2, 1])
        self.assertTrue(np.allclose(expected_out_1, out_epoch_1), 'Out1=\n{}'.format(out_epoch_1))

    def test_shouldLearnScopeAimsMinusInfinityToInfinity(self):
        """Description must be"""
        input_number = output_number = 2
        hidden_layer_number = 3

        great_number = 10000000
        first_sample = np.array([1, 0]).reshape((input_number, 1))
        expected_out = np.array([-great_number, great_number]).reshape((input_number, 1))

        w1 = np.array([1, - 1, 1, 1, -1, 1]).reshape([hidden_layer_number, input_number])
        w2 = np.array([1, - 1, 1, -1, 1, -1]).reshape([output_number, hidden_layer_number])

        mlp_approximator = MlpApproximatorBuilder() \
            .setInputNumber(input_number) \
            .setHiddenLayerNumber(hidden_layer_number) \
            .setOutputNumber(output_number) \
            .setDebugMode(False) \
            .setHiddenLayerWeights(w1) \
            .setOutputLayerWeights(w2) \
            .build()

        mlp_approximator.train(TestingSet([first_sample, expected_out]), 100)
        out_epoch_1 = mlp_approximator.output()
        expected_out_1 = np.array([-8781215.94732426, 8781215.94732425]).reshape([2, 1])
        self.assertTrue(np.allclose(expected_out_1, out_epoch_1), 'Out1=\n{}'.format(out_epoch_1))

    def test_shouldLearnWhenIncreasedHiddenLayerNeurons(self):
        """Description must be"""
        input_number = output_number = 2

        great_number = 10000000
        first_sample = np.array([1, 0]).reshape((input_number, 1))
        expected_out = np.array([-great_number, great_number]).reshape((input_number, 1))

        for hidden_layer_number in range(2, 10):
            mlp_approximator = MlpApproximatorBuilder() \
                .setInputNumber(input_number) \
                .setHiddenLayerNumber(hidden_layer_number) \
                .setOutputNumber(output_number) \
                .setDebugMode(False) \
                .build()

            out_epoch, metrics = mlp_approximator.train(TestingSet([first_sample, expected_out]), 100)
            plt.plot(metrics.Corrections[0])
            plt.xlabel('Epochs (Hidden Neurons={})'.format(hidden_layer_number))
            plt.ylabel('correction')
            plt.show()

            have_same_signs = expected_out * out_epoch >= 0.0
            self.assertTrue(np.alltrue(have_same_signs),
                            '\nOut{}. All fields must have same sign\n{}'
                            .format(hidden_layer_number, have_same_signs))

            delta_expected = np.abs(expected_out - out_epoch)
            error_ratio = np.abs(delta_expected / expected_out)
            accepted_error_level = 0.2
            print('Out{}=\n{}\nDelta_Expected=\n{}\nErrorRatio=\n{}\n'
                  .format(hidden_layer_number, out_epoch, delta_expected, error_ratio))

            self.assertTrue(np.alltrue(accepted_error_level > error_ratio),
                            '\nOut{}=\n{}\nDelta_Expected=\n{}\nErrorRatio=\n{}\n'
                            .format(hidden_layer_number, out_epoch, delta_expected, error_ratio))
