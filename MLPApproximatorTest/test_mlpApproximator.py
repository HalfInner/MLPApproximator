#  Copyright (c) 2020
#  Kajetan Brzuszczak
from contextlib import redirect_stdout
from unittest import TestCase

import numpy as np
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

from MLPApproximator.MlpActivationFunction import TanhActivationFunction, SigmoidActivationFunction
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
        # for max_samples in range(20, 21):
        if True:
            max_samples = 100
            input_number = output_number = 1
            hidden_layer_number = 80
            epoch_number = 30000

            samples = max_samples
            x = np.arange(samples).reshape([samples, 1]) * 2 * np.pi / samples - np.pi
            inputs = np.ascontiguousarray(x, dtype=float)
            # f_x = lambda val: np.sin(val) + 2.
            f_x = lambda x_in: (1 / 20) * (x_in + 4) * (x_in + 2) * (x_in + 1) * (x_in - 1) * (x_in - 3) + 2
            outputs = f_x(inputs)
            outputs = (outputs - np.min(outputs)) / np.ptp(outputs)
            inputs = (inputs - np.min(inputs)) / np.ptp(inputs)

            path = 'C:\\Users\\kajbr\\OneDrive\\Dokumenty\\StudyTmp\\'
            with open('{}Out{:03}.txt'.format(path, max_samples), 'w') as f, redirect_stdout(f):
                mlp_approximator = MlpApproximatorBuilder() \
                    .setInputNumber(input_number) \
                    .setHiddenLayerNumber(hidden_layer_number) \
                    .setOutputNumber(output_number) \
                    .setActivationFunctionForHiddenLayer(TanhActivationFunction()) \
                    .setActivationFunctionForOutputLayer(SigmoidActivationFunction()) \
                    .setDebugMode(True) \
                    .build()

                learned_outputs, metrics = mlp_approximator.train(
                    TestingSet([inputs, outputs]),
                    epoch_number=epoch_number)

                plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[0], 'x-',
                         label='Mean Squared Error')
                plt.xlabel('Epochs={} Samples={} HiddenNeurons={}'.format(epoch_number, samples, hidden_layer_number))
                plt.ylim(0, np.max(metrics.MeanSquaredErrors[0]) * 1.1)
                plt.legend()
                plt.show()
                # plt.savefig('{}{:03}MSE.png'.format(path, max_samples))
                # plt.cla()

                plt.plot(inputs.T[0], outputs.T[0], 'bo', label='True')
                plt.plot(inputs.T[0], learned_outputs.T[0], 'ro-', label='Predicted')
                plt.xlabel('Epochs={} Samples={} HiddenNeurons={}'.format(epoch_number, samples, hidden_layer_number))
                plt.ylim(-0., 1.)
                plt.legend()
                plt.show()
                # plt.savefig('{}{:03}ACC.png'.format(path, max_samples))
                # plt.cla()

    def test_shouldKeras(self):
        test_seed = 1
        np.random.seed(test_seed)
        """
        Description must be

        Noteworthy quote:
            'Normally in my lab the ideal node is around 2 * input * output + 1, e.g., 5-21-2 or 5-11-1,
            but I am not sure this rule of thumb is proven.'
        """

        max_samples = 50
        input_number = output_number = 1
        hidden_layer_number = 100
        epoch_number = 10000

        samples = max_samples
        x = np.arange(samples).reshape([samples, 1]) * 2 * np.pi / samples
        x = x - np.pi
        inputs = np.ascontiguousarray(x, dtype=float)
        # f_x = lambda val: -0.1 * val ** 2 + (1 / 10) * val ** 3.
        f_x = lambda x_in: (1 / 20) * (x_in + 4) * (x_in + 2) * (x_in + 1) * (x_in - 1) * (x_in - 3) + 2
        outputs = f_x(inputs)
        outputs = (outputs - np.min(outputs)) / np.ptp(outputs)
        inputs = (inputs - np.min(inputs)) / np.ptp(inputs)

        if True:
            # the data, split between train and test sets
            x_train = inputs
            y_train = outputs

            model = Sequential()
            model.add(Dense(hidden_layer_number, activation='tanh', input_dim=input_number))
            model.add(Dense(output_number, activation='sigmoid'))
            model.summary()
            model.compile(loss='mae', optimizer='rmsprop', metrics=['mean_squared_error'])

            history = model.fit(x_train, y_train,
                                batch_size=max_samples,
                                epochs=epoch_number,
                                verbose=1,
                                validation_split=0.1)

            print('KEARS: history:\n', history.history['mean_squared_error'])
            plt.plot(np.ascontiguousarray(np.arange(epoch_number)), history.history['mean_squared_error'], 'x-',
                     label='Mean Squared Error')
            plt.xlabel(
                'KERAS: Epochs={} Samples={} HiddenNeurons={}'.format(epoch_number, samples, hidden_layer_number))
            plt.legend()
            plt.ylim(0, np.max(history.history['loss']) * 1.1)
            plt.show()

            predicates = model.predict(inputs)

            print('KEARS: outputs:\n', outputs)
            print('KEARS: predicted:\n', predicates)
            plt.plot(inputs, outputs, 'bo', label="True")
            plt.plot(inputs, predicates, 'ro', label="Predicted")
            plt.xlabel(
                'KERAS: Epochs={} Samples={} HiddenNeurons={}'.format(epoch_number, samples, hidden_layer_number))
            plt.legend()
            plt.ylim(0, 1.)
            plt.show()
