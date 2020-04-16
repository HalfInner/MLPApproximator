import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


class MlpUtils:
    """MLP Utils"""

    def split_data_set(self, input_number, output_number, ratio, required_samples, training_set):
        """

        :param input_number:
        :param ratio:
        :param required_samples:
        :param training_set:
        :return:
        """
        if ratio < 1:
            minimum_required_ratio = 1
            raise ValueError('Ratio must be great or equal to {}'.format(minimum_required_ratio))

        if len(training_set.X.T[0]) != input_number:
            raise ValueError('Input number={} and data set input number={} must be equal'.format(
                input_number, len(training_set.X.T[0])))

        if len(training_set.Y.T[0]) != output_number:
            raise ValueError('Output number={} and data set output number={} must be equal'.format(
                output_number, len(training_set.Y.T[0])))

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
                                   test_outputs, loss, save_to_file):
        """

        :param test_outputs:
        :param file_name:
        :param mlp_approximator:
        :param plot_name:
        :param testing_set_x:
        :param testing_set_y:
        :param save_to_file:
        """

        for idx, test_output in enumerate(test_outputs.T):
            plt.plot(testing_set_x.T[idx], testing_set_y.T[idx], '-', label='F{} Expected'.format(idx))
            plt.plot(testing_set_x.T[idx], test_output, '-',
                     label='F{} Predicted {:2.3}%'.format(idx, loss[idx][0] * 100))
        plt.xlabel('TEST ' + plot_name + ' {:2.3}%'.format(np.mean(loss) * 100))
        plt.ylim(-0.1, 1.1)
        plt.legend()
        if save_to_file:
            plt.savefig('{}_TEST_ACC.png'.format(file_name))
            plt.cla()
        else:
            plt.show()

    def plot_learning_approximation(self, file_name, fitting_set_x, fitting_set_y, learned_outputs, metrics,
                                    plot_name, save_to_file):
        """

        :param file_name:
        :param fitting_set_x:
        :param fitting_set_y:
        :param learned_outputs:
        :param metrics:
        :param plot_name:
        :param save_to_file:
        """

        for idx, learn_output in enumerate(learned_outputs.T):
            plt.plot(fitting_set_x.T[idx], fitting_set_y.T[idx], '-', label='F{} Expected'.format(idx))
            plt.plot(fitting_set_x.T[idx], learn_output, '-',
                     label='F{} Predicted {:2.3}%'.format(idx, metrics.MeanSquaredErrors[idx][-1] * 100))

        plt.xlabel('FIT ' + plot_name + ' {:2.3}%'.format(np.mean(metrics.AvgMeanSquaredError) * 100))

        plt.ylim(-0.1, 1.1)
        plt.legend()
        if save_to_file:
            plt.savefig('{}_FIT_ACC.png'.format(file_name))
            plt.cla()
        else:
            plt.show()

    def plot_rmse(self, epoch_number, file_name, metrics, plot_name, save_to_file):
        """

        :param epoch_number:
        :param file_name:
        :param metrics:
        :param plot_name:
        :param save_to_file:
        :return:
        """
        for idx, mse in enumerate(metrics.MeanSquaredErrors):
            plt.plot(np.ascontiguousarray(np.arange(epoch_number)), mse, '-', label='F{} RMSE'.format(idx))

        plt.plot(np.ascontiguousarray(np.arange(epoch_number)), metrics.AvgMeanSquaredError, 'r-',
                 label='Avg RMSE')
        plt.xlabel('FIT ' + plot_name)
        plt.ylim(0, np.max(metrics.MeanSquaredErrors) * 1.1)
        plt.legend()
        if save_to_file:
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
