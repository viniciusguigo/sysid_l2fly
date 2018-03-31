#!/usr/bin/env python
"""predict.py: use long deep neural nets to predict time sequences for id of
dynamical systems.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "February 08, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.models import model_from_json

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# --------------------------------------


class SysidModel():
    """
    Class that handles model approximation using neural nets.
    """

    def __init__(self):
        self.name = "SysidModel"

        # networks learning parameters
        epochs = 10000
        learning_rate = 0.1
        decay_rate = learning_rate / epochs
        momentum = 0.8
        self.opt = SGD(lr=learning_rate,
                       momentum=momentum,
                       decay=decay_rate,
                       nesterov=False)
        #self.opt = 'adam'
        self.loss = 'mean_squared_error'

        # appending data
        self.append_counter = 0
        self.scale_pqr = 100
        self.dt = 0.005

    def append_and_fit(self, model, s, a):
        """
        Append data generated by the model to original dataset and update current
        model training it again.
        Return updated model
        """
        # convert trajectory to correct format
        s0, a0, s1 = self.data_to_matrix(s, a)
        X, Y = self.aug_matrix(s0, a0)

        # recompile model
        model.compile(loss='mse', optimizer='adam')

        # update model fitting new data
        model.fit(X,
                  Y,
                  nb_epoch=1,
                  batch_size=1,
                  verbose=0)

        # update counter and save model


        return model

    def aug_matrix(self, s, a, n=1):
        """
        Convert time data with states and actions to matrix according to the
        desired 'look back' steps.

        Inputs
        ----------
        s: states
        a: actions
        n: steps to look back

        Outputs
        ----------
        x: past and present states and actions
        y: future states
        """
        # check steps
        assert n >= 1, "n should be >= 1"

        # get dimensions
        samples = s.shape[0]
        features = s.shape[1] + a.shape[1]
        max_rows = samples - n

        # create empty arrays for efficiency
        x = np.zeros((max_rows, n * (features)))
        y = np.zeros((max_rows, s.shape[1]))

        # flat arrays
        data = np.hstack((s, a))
        data_flat = data.flatten()

        # populate y
        j = 0  # to get correct features
        for i in range(max_rows):
            y[i, :] = s[i + n, :]
            x[i, :] = data_flat[j:(j + features * n)]
            j += features

        # normalize the dataset
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = MaxAbsScaler()
        x = self.scaler.fit_transform(x)
        print('x: ', self.scaler.scale_)
        y = self.scaler.fit_transform(y)
        print('y: ', self.scaler.scale_)

        return x, y

    def train_id(self, s0, a0, s1, n, nb_epoch=100, batch_size=10, verbose=2):
        """
        Run deep learning for system id.

        Inputs
        ------
        s0: current state
        a0: control input
        s1: next state
        n: steps back in time
        nb_epoch: number of epochs
        batch_size: batch size for training
        verbose: 0, 1, or 2: amount of print output during training

        Outputs
        -------
        model: trained network model
        history: history of trained model

        """
        # (X) inputs: s0+a0
        # # (Y) outputs: s1
        # X = np.hstack((s0, a0))
        # Y = s1
        X, Y = self.aug_matrix(s0, a0, n)

        # get dimensions
        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        # create model
        model = Sequential()
        # model.add(Dropout(0.2, input_shape=(input_dim,))) # dropout on visible layer (20%)
        model.add(Dense(220,
                        input_shape=(input_dim, ),
                        init='normal',
                        activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(160, init='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(130, init='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, init='normal', activation='linear'))

        # compile model
        model.compile(loss=self.loss, optimizer=self.opt)

        # summary
        print(model.summary())
        print('Number of Inputs: ', input_dim)
        print('Number of Outputs: ', output_dim)

        # fit the model
        history = model.fit(X,
                            Y,
                            nb_epoch=nb_epoch,
                            batch_size=batch_size,
                            verbose=verbose)

        return history, model, X

    def data_to_matrix(self, s, a):
        """
        Convert time data with states and actions to matrix with present states,
        current actions, and future states.

        Inputs
        ----------
        s: states
        a: actions

        Outputs
        ----------
        s0: present states
        a0: present actions
        s1: future states

        """
        # # normalize the dataset
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # s = self.scaler.fit_transform(s)
        # a = self.scaler.fit_transform(a)

        # discard last row of states (no future state to compare)
        s0 = s[0:-1, :]

        # discard first row of actions (normally zero)
        a0 = a[1:]

        # get next states of s0
        s1 = s[1:, :]

        return s0, a0, s1

    def validate_model(self, s, a, n, model):
        """
        Validate model on a new dataset of s and a.

        Inputs
        ----------
        s0: present states
        a0: present actions
        model: network model to validate

        Outputs
        ----------
        s1: future states

        """
        # convert dataset to matrix with n steps looking back
        s0, a0, s1 = self.data_to_matrix(s, a)
        X, Y = self.aug_matrix(s0, a0, n)

        # predict
        # print(X[0:1, :])
        # print(X[0:1, :].shape)
        # Y_pred = model.predict(X[0:1, :])
        # print(Y_pred)
        # print(Y_pred.shape)
        Y_pred = model.predict(X)

        return Y, Y_pred

    def save_model(self, model, name):
        """
        Save neural model to disk.
        """
        # save model to JSON
        model_json = model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)

        # save weights to HDF5
        model.save_weights(name+".h5")
        print("Saved model named "+name+" to disk")

    def load_model(self, name):
        """
        Load neural model from disk.
        """
        # load json and create model
        json_file = open(name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(name+'.h5')
        loaded_model.compile(loss=self.loss, optimizer=self.opt)
        self.model = loaded_model
        print("Loaded model named "+name+" from disk")

        return loaded_model

    def plot_in_out(self, Y, Y_pred, mse, mse2, Y2, Y2_pred, history):
        """
        Plotting train, test, and loss results.
        """

        def convert_to_deg(Y):
            """
            Convert the correct cols to deg.
            """
            Y[:,3:8] = 180/np.pi*Y[:,3:8]
            Y[:,9:12] = 180/np.pi*Y[:,9:12]
            return Y

        ## =====================
        # load stylesheet
        try:
            plt.style.use("dwplot")
        except:
            print("Cannot use this stylesheet")

        # labels
        data_label = np.array([['u [ft/s]', 'v [ft/s]', 'w [ft/s]'],
                               ['p [deg/s]', 'q [deg/s]', 'r [deg/s]'],
                               ['latitude [deg]', 'longitude [deg]', 'altitude [ft]'],
                               ['roll [deg]', 'pitch [deg]', 'yaw [deg]']])

        data_title = np.array(['Linear Velocity (Training Data)',
                               'Angular Velocity (Training Data)',
                               'Position (Training Data)',
                               'Orientation - Euler Angles (Training Data)',
                               'Linear Velocity (Test Data)',
                               'Angular Velocity (Test Data)',
                               'Position (Test Data)',
                               'Orientation - Euler Angles (Test Data)'])

        save_figs = []

        # calculate time vector
        time_train = np.arange(0,len(Y))*self.dt
        time_test = np.arange(0,len(Y2))*self.dt

        # re-scale
        Y = self.scaler.inverse_transform(Y)
        Y_pred = self.scaler.inverse_transform(Y_pred)
        Y2 = self.scaler.inverse_transform(Y2)
        Y2_pred = self.scaler.inverse_transform(Y2_pred)

        # convert from rad to deg
        Y = convert_to_deg(Y)
        Y_pred = convert_to_deg(Y_pred)
        Y2 = convert_to_deg(Y2)
        Y2_pred = convert_to_deg(Y2_pred)


        # plot (training and testing)
        # 4 figures with 3 subplots each:
        #   - linear vel
        #   - angular vel
        #   - position
        #   - euler angles
        for k in range(2):
            # select data
            if k == 0: # train
                data_Y = Y
                data_Y_pred = Y_pred
                data_mse = mse

            elif k == 1: # test
                data_Y = Y2
                data_Y_pred = Y2_pred
                data_mse = mse2

            # time vectors
            time_vec = np.arange(0,len(data_Y))*self.dt

            for j in range(4):
                fig = plt.figure(j + k*4)
                plt.suptitle(data_title[j + 4*k], fontsize='medium')

                for col in range(3):
                    data_index = col + 3*j
                    index = 311 + col
                    ax = fig.add_subplot(index)
                    ax.plot(time_vec, data_Y[:, data_index], 'b', label='Truth')
                    ax.plot(time_vec, data_Y_pred[:, data_index], 'r--', label='Predicted')
                    ax.set_xlim([0, time_vec[-1]])
                    ax.grid()
                    # plt.tight_layout()
                    # ax.legend(loc='best')

                    # plot settings
                    x_text = .74
                    y_text_bias = -.15
                    if index == 311:
                        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                  loc=4,
                                  ncol=2,
                                  borderaxespad=0.)
                        ax.set_ylabel(data_label[j,col])
                        plt.figtext(x_text,
                                 1 + y_text_bias,
                                 'MSE = %.4f' % data_mse[data_index],
                                 weight='medium')
                    elif index == 312:
                        ax.set_ylabel(data_label[j,col])
                        plt.figtext(x_text,
                                 .725 + y_text_bias,
                                 'MSE = %.4f' % data_mse[data_index],
                                 weight='medium')
                    elif index == 313:
                        ax.set_ylabel(data_label[j,col])
                        ax.set_xlabel('Time [sec]')
                        plt.figtext(x_text,
                                 .45 + y_text_bias,
                                 'MSE = %.4f' % data_mse[data_index],
                                 weight='medium')

                # append each figure with 3 subplots
                save_figs.append(fig)

        ## =====================
        # plot (loss)
        fig_loss = plt.figure()
        #plt.suptitle('Training Loss', fontsize='medium')
        plt.xlabel('Epoch [unit]')
        plt.ylabel('Mean Squared Error [unit]')
        plt.grid()

        hist_list = history.history['loss']
        hist_init = int(0.0 * len(
            hist_list))  # cut first x% of the history (for scaling purposes)
        plt.plot(hist_list[hist_init:])
        plt.xlim([0, len(hist_list)])

        save_figs.append(fig_loss)

        # save plots
        count_fig = 0
        fig_title = ['train_vel',
                     'train_ang',
                     'train_pos',
                     'train_eul',
                     'test_vel',
                     'test_ang',
                     'test_pos',
                     'test_eul',
                     'loss']

        for each_fig in save_figs:
            fig_name = './results/'+ fig_title[count_fig] + '.png'
            each_fig.savefig(fig_name,dpi=300,transparent=True)
            count_fig += 1
        print('Figures saved.')

    def fit_model(self):
        """
        Lazy way of fitting the model.
        """
        # fix random seed for reproducibility
        np.random.seed(42)

        # load the dataset
        # dataset_s = np.loadtxt("../data/Smoothed_Data/out_latD_smooth_all.csv",
        #                        delimiter=",")
        dataset_s = np.loadtxt("../data/gtm/states_train.csv",
                               delimiter=",")
        s = dataset_s[:, 0:12]

        #dataset_a2 = np.loadtxt("../data/data_in_test.csv", delimiter=",")
        # dataset_a = np.loadtxt("../data/Smoothed_Data/in_latD_smooth_all.csv",
        #                        delimiter=",")
        dataset_a = np.loadtxt("../data/gtm/doublet_sent_train.csv",
                               delimiter=",")
        a = dataset_a[:, 0:3]

        # convert dataset to matrix with n steps looking back
        s0, a0, s1 = self.data_to_matrix(s, a)

        # # normalize data
        # X = preprocessing.normalize(X,Y, Y_pred, mse, mse2, Y2, Y2_pred, history norm='l2')
        # Y = preprocessing.normalize(Y, norm='l2')

        # train network
        n = 1  # steps back in time
        history, model, X_check = self.train_id(s0,
                                                a0,
                                                s1,
                                                n,
                                                nb_epoch=200,
                                                batch_size=50,
                                                verbose=2)
        self.model = model

        # normalize
        # normalize the dataset
        X, Y = self.aug_matrix(s0, a0, n)

        Y_pred = model.predict(X)

        ## =====================
        # testing more data
        # load the dataset
        #dataset_s2 = np.loadtxt("../data/data_out_test.csv", delimiter=",")
        dataset_s2 = np.loadtxt("../data/gtm/states_test.csv",
                                delimiter=",")
        s2 = dataset_s2[:, 0:12]

        dataset_a2 = np.loadtxt("../data/gtm/doublet_sent_test.csv",
                                delimiter=",")
        a2 = dataset_a2[:, 0:3]

        # validate model
        Y2, Y2_pred = self.validate_model(s2, a2, n, model)

        # compute MSE for each column (feature)
        mse2 = ((Y2 - Y2_pred)**2).mean(axis=0)
        print('Total MSE error (test) = ', sum(mse2))

        # compute MSE for each column (feature)
        mse = ((Y - Y_pred)**2).mean(axis=0)
        print('Total MSE error (training) = ', sum(mse))

        # save model and plot
        self.save_model(model, 'model')
        self.plot_in_out(Y, Y_pred, mse, mse2, Y2, Y2_pred, history)


if __name__ == "__main__":
    sysid_model = SysidModel()
    sysid_model.fit_model()