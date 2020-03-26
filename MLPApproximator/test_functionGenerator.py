#  Copyright (c) 2020
#  Kajetan Brzuszczak

from unittest import TestCase

from MLPApproximator.MlpAssembly import FunctionGenerator


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
        self.assertEqual([0], testing_set.X[first_function_index])
        self.assertEqual([c], testing_set.Y[first_function_index])

    def test_generateOneSampleDoubleFunction(self):
        function_generator = FunctionGenerator()
        a, b, c = 10, 4, 3
        function_generator.addFunction([a, b, c])
        function_generator.addFunction([a, b, c + 1])
        testing_set = function_generator.generate()

        first_sample_indxe = 0
        first_function_index = 0
        self.assertEqual(0, testing_set.X[first_function_index][first_sample_indxe])
        self.assertEqual(c, testing_set.Y[first_function_index][first_sample_indxe])

        second_function_index = 1
        self.assertEqual(0, testing_set.X[second_function_index][first_sample_indxe])
        self.assertEqual(c + 1, testing_set.Y[second_function_index][first_sample_indxe])

    def test_generateTrippleSampleTrippleFunction(self):
        function_generator = FunctionGenerator()
        function_number = 3
        base_factor = 10
        for x in range(function_number):
            # f(x) = 2x^1 + 10
            function_generator.addFunction([2, base_factor])

        sample_number = 3
        testing_set = function_generator.generate(sample_number)

        for function_idx in range(function_number):
            for sample_idx in range(sample_number):
                self.assertEqual(2 * sample_idx + base_factor, testing_set.Y[function_idx][sample_idx])
