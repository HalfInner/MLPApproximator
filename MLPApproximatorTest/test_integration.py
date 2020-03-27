#  Copyright (c) 2020
#  Kajetan Brzuszczak
from datetime import date
from unittest import TestCase


class TestIntegration(TestCase):
    """
    Draft description for research:
        1) generate learning dataset - doubled - training/testing
        2) teach neural network three inputs three outputs
            3 different function with M={3,5,7} parameters
        3) for each tipple function group conduct tests - training and testing
            a) for each test change hidden layer nodes from N=M to N=10*M.
                b) for each hidden layer(?) conduct tests where you have I=100 epochs of learning heading to 1000
        4) process results for each combination
    """

    def test_runFromDeadline(self):
        delta = date(2020, 0o4, 0o3) - date.today()
        self.assertTrue(delta.days >= 0, 'Not this time my friend, night is long... And you\'ve missed your deadline')
