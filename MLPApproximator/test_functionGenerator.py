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

        expected_output = '10x^2 + 4x + 3\n'\
                          '-5x^2 + 4x + 3\n'
        self.assertEqual(expected_output, function_generator.to_string())
