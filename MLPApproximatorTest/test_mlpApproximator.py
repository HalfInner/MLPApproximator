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
        Book example: One complete iteration o learning 3-layers perceptron (MLP).
        3 neurons in hidden layer. 2 neuron on input, and 2 neurons on output
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

        for hidden_layer_number in range(2, 30):
            mlp_approximator = MlpApproximatorBuilder() \
                .setInputNumber(input_number) \
                .setHiddenLayerNumber(hidden_layer_number) \
                .setOutputNumber(output_number) \
                .setDebugMode(False) \
                .build()
            epoch_number = 100
            out_epoch, metrics = mlp_approximator.train(TestingSet([first_sample, expected_out]), epoch_number)
            # plt.plot(metrics.Corrections[0], label='Correction Out1')
            # plt.plot(metrics.Corrections[1], label='Correction Out2')
            plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[0],
                     label='Mean Squared Error Out1')
            plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[1],
                     label='Mean Squared Error Out2')
            plt.plot(np.mean(metrics.MeanSquaredErrors, axis=0), label='Mean Squared Error AVG')
            plt.xlabel('Epochs (Hidden Neurons={})'.format(hidden_layer_number))
            plt.legend()
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

    def test_shouldLearnWhenIncreasedNumberOfSamples(self):
        test_seed = 1
        np.random.seed(test_seed)
        """
        Description must be

        Noteworthy quote:
            'Normally in my lab the ideal node is around 2 * input * output + 1, e.g., 5-21-2 or 5-11-1,
            but I am not sure this rule of thumb is proven.'
        """

        max_samples = 3
        input_number = output_number = 1
        hidden_layer_number = 2

        # for samples in range(2, max_samples + 1):
        # for samples in range(max_samples, max_samples + 1):
        samples = max_samples
        if True:
            mlp_approximator = MlpApproximatorBuilder() \
                .setInputNumber(input_number) \
                .setHiddenLayerNumber(hidden_layer_number) \
                .setOutputNumber(output_number) \
                .setDebugMode(True) \
                .build()

            x = np.arange(samples).reshape([samples, 1]) * np.pi / samples
            inputs = np.ascontiguousarray(x, dtype=float)
            f_x = lambda val: np.sin(val) + 0.5
            # f_x = lambda val: val + 1
            outputs = f_x(inputs)

            epoch_number = 2000

            learned_outputs, metrics = mlp_approximator.train(
                TestingSet([inputs, outputs]),
                epoch_number=epoch_number)

            # plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[0], 'x',
            #          label='Mean Squared Error')
            plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors, 'x',
                     label='Mean Squared Error')
            plt.xlabel('Epochs={} Samples={} HiddenNeurons={}'.format(epoch_number, samples, hidden_layer_number))
            plt.ylim(0, np.max(metrics.MeanSquaredErrors[0]) * 1.1)
            plt.legend()
            plt.show()

            plt.plot(outputs.T[0], 'x-', label='Out')
            plt.plot(learned_outputs[0], 'x-', label='Approximation')
            plt.xlabel('Epochs={} Samples={} HiddenNeurons={}'.format(epoch_number, samples, hidden_layer_number))
            plt.ylim(0, max(outputs.T[0]) + 1.1)
            plt.legend()
            plt.show()

            have_same_signs = outputs * learned_outputs >= 0.0
            self.assertTrue(np.alltrue(have_same_signs),
                            '\nOut{}. All fields must have same sign\n{}'
                            .format(hidden_layer_number, have_same_signs))

            metrics.MeanSquaredErrors
            accepted_error_level = 0.4
            print('Out{}=\n{}\n\nErrorRatio=\n{}\n'
                  .format(epoch_number, learned_outputs, metrics.MeanSquaredErrors))

            self.assertTrue(np.alltrue(accepted_error_level > metrics.MeanSquaredErrors),
                            '\nOut=\n{}\nErrorRatio=\n{}\n'
                            .format(learned_outputs, metrics.MeanSquaredErrors))
