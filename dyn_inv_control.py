#!/usr/bin/env python
"""dyn_inv_control.py: dynamic inversion control.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "April 25, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


class DI_Control():
    """
    Dynamic Inversion Control class for dynamical systems approximated using
    neural networks.
    """

    def __init__(self):
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

        # controller parameters
        self.delta = 1e-3  # control perturbation
        self.dt = 0.005

    def load_model(self, name):
        """
        Load neural network model (using Keras: h5 and json files).

        Inputs
        ---------
        name: name of the model appended with address.
        """
        # load json and create model
        json_file = open(name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(name + '.h5')
        loaded_model.compile(loss=self.loss, optimizer=self.opt)
        self.model = loaded_model
        print("Loaded model named " + name + " from disk")

    def plant_dynamics(self, x, u):
        """
        Simulate a single time step of the plant, from
        initial state x and applying control signal u

        x np.array: the state of the system
        u np.array: the control signal
        """
        # do a feedforward pass on the network (model)
        x_in = np.hstack((x, u)).reshape(1, x.shape[0] + u.shape[0])

        # normalize inputs and predict
        x_in = self.scaler.transform(x_in)
        xnext = self.model.predict_on_batch(x_in)

        # re-scale
        xnext = self.scaler_y.inverse_transform(xnext)

        return xnext

    def compute_delta_u(self, x, u, num_controls):
        """
        Compute finite difference for each control.
        """
        # preserve initial u
        u_plus = np.copy(u)
        u_minus = np.copy(u)
        g = np.zeros((num_controls, num_controls))

        # compute perturbation for each control
        for idx in range(num_controls):
            # perturb control
            u_plus[idx] = u[idx] + self.delta
            u_minus[idx] = u[idx] - self.delta

            # compute dynamics
            dyn_plus = self.plant_dynamics(x, u_plus)
            dyn_minus = self.plant_dynamics(x, u_minus)

            # select only feedbacked states to compute g
            partial_g = 1 / (2 * self.delta) * (dyn_plus[0,3:6] - dyn_minus[0,3:6])
            g[:,idx] = partial_g

        # compute delta_u
        f_bar = self.plant_dynamics(x, u)
        delta_u = np.linalg.pinv(g) @ (self.xd - f_bar[0,3:6].reshape(num_controls,1))

        return delta_u.reshape(3)

    def use_doublet(self, i, n_steps, num_controls):
        """
        Send spaced doublets to the plant.
        """
        # 1st control
        if (i > .10*n_steps) and (i < .11*n_steps):
            u = np.array([1,0,0]).reshape(num_controls)
        elif (i > .11*n_steps) and (i < .12*n_steps):
            u = np.array([-1,0,0]).reshape(num_controls)

        # 2nd control
        elif (i > .30*n_steps) and (i < .31*n_steps):
            u = np.array([0,1,0]).reshape(num_controls)
        elif (i > .31*n_steps) and (i < .32*n_steps):
            u = np.array([0,-1,0]).reshape(num_controls)

        # 3rd control
        elif (i > .50*n_steps) and (i < .51*n_steps):
            u = np.array([0,0,1]).reshape(num_controls)
        elif (i > .51*n_steps) and (i < .52*n_steps):
            u = np.array([0,0,-1]).reshape(num_controls)

        else:
            u = np.zeros((num_controls, 1)).reshape(num_controls)

        return u


    def test_controller(self):
        """
        Load model from disk and test controller.
        """
        # load model
        name = './models/model'
        self.load_model(name)  # saved at controller.model

        # define inital parameters
        n_steps = 1000
        num_states = 12
        num_controls = 3

        # define trajectory and initial conditions
        self.xd = np.array([.01,0.,0.]).reshape(3,1)
        x0 = np.array([164, 0, 8.5, 0, 0, 0, 0.6, -1.3, 800, 0, .05,
                       1.5]).reshape(num_states, 1)

        u0 = np.zeros((num_controls, 1))

        # create vectors to save data and append inital states
        xsim = np.zeros((n_steps, num_states))
        usim = np.zeros((n_steps, num_controls))

        xsim[0, :] = x0.reshape(num_states)
        usim[0, :] = u0.reshape(num_controls)

        # normalize initial conditions so neural net can work
        self.scaler = MaxAbsScaler()
        self.scaler.scale_ = np.array(
            [1.64200000e+02, 5.35480000e+00, 1.22180000e+01, 3.79070000e-01,
             1.68410000e-01, 1.69720000e-01, 6.60220000e-01, 1.31760000e+00,
             8.00000000e+02, 7.72520000e-02, 6.82860000e-02, 1.60590000e+00,
             1.00000000e+00, 1.00000000e+00, 1.00000000e+00])

        self.scaler_y = MaxAbsScaler()
        self.scaler_y.scale_ = np.array([1.64200000e+02,
                                         5.35750000e+00,
                                         1.22180000e+01,
                                         3.79270000e-01,
                                         1.68410000e-01,
                                         1.70610000e-01,
                                         6.60220000e-01,
                                         1.31760000e+00,
                                         8.00000000e+02,
                                         7.40890000e-02,
                                         6.82860000e-02,
                                         1.59430000e+00, ])

        # simulate plant dynamics
        for i in range(1, n_steps):
            # # apply control to plant
            # xnext = self.plant_dynamics(xsim[i - 1, :], usim[i - 1, :])
            # xsim[i, :] = xnext.reshape(num_states)
            # usim[i, :] = self.use_doublet(i, n_steps, num_controls)

            # states
            x = xsim[i - 1, :]
            u1 = usim[i - 1, :]

            # compute control and bound it
            delta_u = self.compute_delta_u(x, u1, num_controls)
            u = np.clip(u1 + delta_u, -1, 1)

            # apply control to plant
            xnext = self.plant_dynamics(x, u)
            xsim[i , :] = xnext
            usim[i , :] = u

        # plot
        self.plot_controller(xsim, usim)

    def plot_controller(self, x, u):
        """
        Plot simulated states and controls.
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
                               ['roll [deg]', 'pitch [deg]', 'yaw [deg]'],
                               ['delta_a [deg]', 'delta_e [deg]', 'delta_r [deg]']])

        data_title = np.array(['Linear Velocity',
                               'Angular Velocity',
                               'Position',
                               'Orientation - Euler Angles',
                               'Control'])

        save_figs = []

        # convert from rad to deg
        x = convert_to_deg(x)

        # stack for plotting
        x = np.hstack((x,u))

        # plot (training and testing)
        # 4 figures with 3 subplots each:
        #   - linear vel
        #   - angular vel
        #   - position
        #   - euler angles

        # time vectors
        time_vec = np.arange(0,len(x))*self.dt

        for j in range(5):
            fig = plt.figure(j)
            plt.suptitle(data_title[j], fontsize='medium')

            # plot states
            for col in range(3):
                data_index = col + 3*j
                index = 311 + col
                ax = fig.add_subplot(index)
                ax.plot(time_vec, x[:, data_index], 'b')
                if j == 1:
                    xd_line = np.ones((len(x),1))*180/np.pi*self.xd[col]
                    ax.plot(time_vec, xd_line, '--r', label='xd')
                ax.set_xlim([0, time_vec[-1]])
                ax.grid()
                # plt.tight_layout()
                # ax.legend(loc='best')

                # plot settings
                if index == 311:
                    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                              loc=4,
                              ncol=2,
                              borderaxespad=0.)
                    ax.set_ylabel(data_label[j,col])

                elif index == 312:
                    ax.set_ylabel(data_label[j,col])

                elif index == 313:
                    ax.set_ylabel(data_label[j,col])
                    ax.set_xlabel('Time [sec]')


            # append each figure with 3 subplots
            save_figs.append(fig)


        plt.show()


if __name__ == "__main__":
    controller = DI_Control()
    controller.test_controller()
