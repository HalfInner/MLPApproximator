#  Copyright (c) 2020
#  Kajetan Brzuszczak

from unittest import TestCase

import numpy as np

from MLPApproximator.MlpFunctionGenerator import FunctionGenerator


class TestFunctionGenerator(TestCase):
    def test_addOneFunction(self):
        function_generator = FunctionGenerator()
        function_generator.addFunction([10, 4, 3])

        expected_output = "10x^2 + 4x + 3\n"
        self.assertEqual(expected_output, function_generator.to_string())

    def test_addTwoFunctions(self):
        function_generator = FunctionGenerator()
        function_generator.addFunction([10, 4, 3])
        function_generator.addFunction([-5, 4, 3])

        expected_output = '10x^2 + 4x + 3\n' \
                          '-5x^2 + 4x + 3\n'
        self.assertEqual(expected_output, function_generator.to_string())

    def test_generateOneSample(self):
        function_generator = FunctionGenerator()
        a, b, c = 10, 4, 3
        function_generator.addFunction([a, b, c])
        testing_set = function_generator.generate()

        first_function_index = 0
        self.assertEqual([-1], testing_set.X[first_function_index])
        self.assertEqual([0.5], testing_set.Y[first_function_index])

    def test_generateOneSampleDoubleFunction(self):
        function_generator = FunctionGenerator()
        a, b, c = 10., 4., 3.
        function_generator.addFunction([a, b, c])
        function_generator.addFunction([a, b, c + 1.])
        testing_set = function_generator.generate()

        first_sample_indxe = 0
        first_function_index = 0
        self.assertAlmostEqual(-1., testing_set.X[first_function_index][first_sample_indxe])
        self.assertAlmostEqual(0.5, testing_set.Y[first_function_index][first_sample_indxe])

        second_function_index = 1
        self.assertAlmostEqual(0., testing_set.X[second_function_index][first_sample_indxe])
        self.assertAlmostEqual(0.5, testing_set.Y[second_function_index][first_sample_indxe])

    def test_generateTripleSampleTripleFunction(self):
        function_generator = FunctionGenerator()
        function_number = 3
        base_factor = 10
        for x in range(function_number):
            # f(x) = 2x^1 + 10
            function_generator.addFunction([2, base_factor])

        sample_number = 3
        testing_set = function_generator.generate(sample_number)

        for function_idx in range(function_number):
            self.assertTrue(np.all(np.array([0, 0.5, 1] == testing_set.Y[function_idx])))

    def test_generate100ContinuousSamples(self):
        function_generator = FunctionGenerator()
        # f(x) = x
        function_generator.addFunction([1, 0])

        sample_number = 100
        testing_set = function_generator.generate(sample_number)

        first_function_idx = 0
        for sample_idx in range(sample_number):
            self.assertAlmostEqual(sample_idx/sample_number, testing_set.Y[first_function_idx][sample_idx], delta=0.011)
