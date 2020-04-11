import numpy as np


class MLPMetrics:
    """Metrics container"""

    def __init__(self) -> None:
        self.__corrections = None
        self.__mean_squared_errors = None

    @property
    def Corrections(self):
        return self.__corrections

    @property
    def MeanSquaredErrors(self):
        return self.__mean_squared_errors

    def addCorrection(self, correction: np.array):
        if self.__corrections is None:
            self.__corrections = correction
            return

        self.__corrections = np.append(self.__corrections, correction, axis=1)

    def addMeanSquaredError(self, mean_squared_error: np.array):
        if self.__mean_squared_errors is None:
            self.__mean_squared_errors = mean_squared_error
            return

        self.__mean_squared_errors = np.append(self.__mean_squared_errors, mean_squared_error, axis=1)
