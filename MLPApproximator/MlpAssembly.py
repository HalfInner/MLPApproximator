#  Copyright (c) 2020
#  Kajetan Brzuszczak

import numpy as np

from MLPApproximator.MlpApproximator import MlpApproximator


class MlpApproximatorAssembler:
    """
    Assembly modules from the project

    example::
        def run(self) -> str:
            first_sample = np.array([1, 2]).reshape((2, 1))
            expected_out = np.array([1, 0]).reshape((2, 1))

            mlp_approximator = MlpApproximator(2, 2, 3)
            mlp_approximator.propagateForward(first_sample)
            mlp_approximator.propagateErrorBackward(expected_out)

            return 0
         ``
    """

    def run(self) -> str:
        """
        :return: error code 0-OK
        """
        # sample: x=(1,2); output: f(x) = (1, 0).
        first_sample = np.array([1, 2]).reshape((2, 1))
        expected_out = np.array([1, 0]).reshape((2, 1))

        mlp_approximator = MlpApproximator(2, 2, 3)
        mlp_approximator.propagateForward(first_sample)
        mlp_approximator.propagateErrorBackward(expected_out)

        return 0
