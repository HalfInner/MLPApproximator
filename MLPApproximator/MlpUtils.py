import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


class MlpUtils:
    """MLP Utils"""

    def split_data_set(self, input_number, ratio, required_samples, training_set):
        """

        :param input_number:
        :param ratio:
        :param required_samples:
        :param training_set:
        :return:
        """
        fitting_set_x = np.empty((0, input_number))
        fitting_set_y = np.empty((0, input_number))
        testing_set_x = np.empty((0, input_number))
        testing_set_y = np.empty((0, input_number))
        for idx in range(required_samples):
            if idx % ratio:
                fitting_set_x = np.append(fitting_set_x, np.array([training_set.X.T[idx]]), axis=0)
                fitting_set_y = np.append(fitting_set_y, np.array([training_set.Y.T[idx]]), axis=0)
            else:
                testing_set_x = np.append(testing_set_x, np.array([training_set.X.T[idx]]), axis=0)
                testing_set_y = np.append(testing_set_y, np.array([training_set.Y.T[idx]]), axis=0)
        return fitting_set_x, fitting_set_y, testing_set_x, testing_set_y

    def plot_testing_approximation(self, file_name, plot_name, testing_set_x, testing_set_y,
                                   test_output, loss, to_file):
        """

        :param test_output:
        :param file_name:
        :param mlp_approximator:
        :param plot_name:
        :param testing_set_x:
        :param testing_set_y:
        :param to_file:
        """
        plt.plot(testing_set_x.T[0], testing_set_y.T[0], 'b-', label='F1 Expected')
        plt.plot(testing_set_x.T[0], test_output.T[0], 'y-',
                 label='F1 Predicted {:2.3}%'.format(loss[0][0] * 100))
        plt.plot(testing_set_x.T[1], testing_set_y.T[1], 'g-', label='F2 Expected')
        plt.plot(testing_set_x.T[1], test_output.T[1], 'r-',
                 label='F2 Predicted {:2.3}%'.format(loss[1][0] * 100))
        plt.plot(testing_set_x.T[2], testing_set_y.T[2], 'k-', label='F3 Expected')
        plt.plot(testing_set_x.T[2], test_output.T[2], 'm-',
                 label='F3 Predicted {:2.3}%'.format(loss[2][0] * 100))
        plt.xlabel('TEST ' + plot_name + ' {:2.3}%'.format(np.mean(loss) * 100))
        plt.ylim(-0.1, 1.1)
        plt.legend()
        if to_file:
            plt.savefig('{}_TEST_ACC.png'.format(file_name))
            plt.cla()
        else:
            plt.show()

    def plot_learning_approximation(self, file_name, fitting_set_x, fitting_set_y, learned_outputs, metrics,
                                    plot_name, to_file):
        """

        :param file_name:
        :param fitting_set_x:
        :param fitting_set_y:
        :param learned_outputs:
        :param metrics:
        :param plot_name:
        :param to_file:
        """
        plt.plot(fitting_set_x.T[0], fitting_set_y.T[0], 'b-', label='F1 Expected')
        plt.plot(fitting_set_x.T[0], learned_outputs.T[0], 'y-', label='F1 Predicted')
        plt.plot(fitting_set_x.T[1], fitting_set_y.T[1], 'g-', label='F2 Expected')
        plt.plot(fitting_set_x.T[1], learned_outputs.T[1], 'r-', label='F2 Predicted')
        plt.plot(fitting_set_x.T[2], fitting_set_y.T[2], 'k-', label='F3 Expected')
        plt.plot(fitting_set_x.T[2], learned_outputs.T[2], 'm-', label='F3 Predicted')
        plt.xlabel('FIT ' + plot_name + ' {:2.3}%'.format(np.mean(metrics.AvgMeanSquaredError) * 100))
        plt.ylim(-0.1, 1.1)
        plt.legend()
        if to_file:
            plt.savefig('{}_FIT_ACC.png'.format(file_name))
            plt.cla()
        else:
            plt.show()

    def plot_rmse(self,  epoch_number, file_name, metrics, plot_name, to_file):
        """

        :param epoch_number:
        :param file_name:
        :param hidden_layer_number:
        :param metrics:
        :param parameter_m:
        :param sub_test_idx:
        :param to_file:
        :return:
        """
        plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[0], 'b-',
                 label='F1 RMSE')
        plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[1], 'r-',
                 label='F2 RMSE')
        plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.MeanSquaredErrors[2], 'g-',
                 label='F3 RMSE')
        plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.AvgMeanSquaredError, 'm-',
                 label='Avg RMSE')
        plt.xlabel('FIT ' + plot_name)
        plt.ylim(0, np.max(metrics.MeanSquaredErrors) * 1.1)
        plt.legend()
        if to_file:
            plt.savefig('{}_FIT_MSE.png'.format(file_name))
            plt.cla()
        else:
            plt.show()
        return plot_name

    def create_date_folder_if_not_exists(self, base_directory='..\\TestResults\\'):
        """

        :return:
        """
        today = datetime.now()
        folder_name = today.strftime('%Y%m%H%M')
        directory = base_directory + folder_name + '\\'
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
