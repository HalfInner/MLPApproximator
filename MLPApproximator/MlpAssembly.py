import numpy as np

from MLPApproximator.MlpApproximator import MlpApproximator


class MlpApproximatorTester:
    def run(self) -> str:
        # sample: x=(1,2); output: f(x) = (1, 0).
        first_sample = np.array([1, 2]).reshape((2, 1))
        expected_out = np.array([1, 0]).reshape((2, 1))

        mlp_approximator = MlpApproximator(2, 2)
        mlp_approximator.forwardPropagation(first_sample)
        mlp_approximator.doWeirdStuff(expected_out)

        return 0
