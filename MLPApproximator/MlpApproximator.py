from MLPApproximator.MlpPerceptron import Perceptron


class MlpApproximator:

    def __init__(self, input_number, output_number, hidden_layer_number=3):
        self.__input_number = input_number
        self.__output_number = output_number
        self.__p1 = Perceptron(input_number, hidden_layer_number)
        self.__p2 = Perceptron(hidden_layer_number, output_number, test_first_p1=False)

    def forwardPropagation(self, input_data):
        self.__p2.forwardPropagation(self.__p1.forwardPropagation(input_data).output())

    def doWeirdStuff(self, output_data):
        self__mean_square_error = self.__p2.meanSquaredError(output_data)
        print('P2: mean error \n', self__mean_square_error)