#  Copyright (c) 2020
#  Kajetan Brzuszczak
import numpy as np

from MLPApproximator.MlpActivationFunction import SigmoidActivationFunction
from MLPApproximator.MlpApproximator import MlpApproximator


class MlpApproximatorBuilder:
    """Builder"""

    def __init__(self) -> None:
        self.__input_number = None
        self.__output_number = None
        self.__hidden_layer_number = None
        self.__activation_function_hidden_layer = None
        self.__activation_function_output_layer = None
        self.__debug_mode = None
        self.__hidden_layer_weights = None
        self.__output_layer_weights = None

    def setInputNumber(self, input_number: int):
        self.__input_number = input_number
        return self

    def setOutputNumber(self, output_number: int):
        self.__output_number = output_number
        return self

    def setHiddenLayerNumber(self, hidden_layer_number: int):
        self.__hidden_layer_number = hidden_layer_number
        return self

    def setActivationFunctionForHiddenLayer(self, activation_function):
        self.__activation_function_hidden_layer = activation_function
        return self

    def setActivationFunctionForOutputLayer(self, activation_function):
        self.__activation_function_output_layer = activation_function
        return self

    def setDebugMode(self, debug_mode: bool):
        self.__debug_mode = debug_mode
        return self

    def setHiddenLayerWeights(self, weights: np.array):
        self.__hidden_layer_weights = weights
        return self

    def setOutputLayerWeights(self, weights: np.array):
        self.__output_layer_weights = weights
        return self

    def build(self):
        if self.__input_number is None:
            raise RuntimeError('Input node number is required')
        if self.__output_number is None:
            raise RuntimeError('Output node number is required')
        if self.__hidden_layer_number is None:
            raise RuntimeError('Hidden layer node number is required')
        if self.__activation_function_hidden_layer is None:
            self.__activation_function_hidden_layer = SigmoidActivationFunction()
        if self.__activation_function_output_layer is None:
            self.__activation_function_output_layer = SigmoidActivationFunction()
        if self.__debug_mode is None:
            self.__debug_mode = False
        if self.__hidden_layer_weights is None:
            pass
        if self.__output_layer_weights is None:
            pass

        return MlpApproximator(
            input_number=self.__input_number,
            output_number=self.__output_number,
            hidden_layer_number=self.__hidden_layer_number,
            activation_function_hidden_layer=self.__activation_function_hidden_layer,
            activation_function_output_layer=self.__activation_function_output_layer,
            debug_level_1_on=self.__debug_mode,
            hidden_layer_weights=self.__hidden_layer_weights,
            output_layer_weights=self.__output_layer_weights)
