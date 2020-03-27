#  Copyright (c) 2020
#  Kajetan Brzuszczak
import sys

import numpy as np
from io import StringIO
from unittest import TestCase

from MLPApproximator.MlpApproximator import MlpApproximator


class TestMlpApproximator(TestCase):

    def test_properInitialization(self):
        mlp_approximator = MlpApproximator(2, 2, 3, debug_on=True)
        self.assertEqual(None, mlp_approximator.meanSquaredError())

    def test_propagateForward(self):
        """
        Not proper test, but makes sure that changes implementation won't influence on already verified basic  behaviour
        """
        first_sample = np.array([1, 2]).reshape((2, 1))
        expected_out = np.array([1, 0]).reshape((2, 1))

        original_stdout = sys.stdout
        sys.stdout = string_io_out = StringIO()

        mlp_approximator = MlpApproximator(2, 2, 3, debug_on=True)
        mlp_approximator.propagateForward(first_sample)
        mlp_approximator.propagateErrorBackward(expected_out)

        sys.stdout = original_stdout

        expected_out = """
            Weights 
             [[1.00026546 1.00026546 1.00026546]
             [0.99537525 0.99537525 0.99537525]]
            Next Weight 
             [[1.00026546 1.00026546 1.00026546]
             [0.99537525 0.99537525 0.99537525]]
            next_mean_squared_error 
             [[ 0.00278674]
             [-0.04855007]]
            Sq : 
             [[-0.00205726]
             [-0.00205726]
             [-0.00205726]]
            Weights 
             [[0.99979427 0.99958855]
             [0.99979427 0.99958855]
             [0.99979427 0.99958855]]
        """
        self.assertEqual(expected_out.replace(' ', '').replace('\n', ''),
                         string_io_out.getvalue().replace(' ', '').replace('\n', ''))
