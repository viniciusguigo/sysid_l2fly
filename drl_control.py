#!/usr/bin/env python
"""drl_control.py: test controller on learned model.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "April 17, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import LSTM

from predict import SysidModel
from control import Neural_iLQR

def plant_dynamics(x, u, model):
    """
    Simulate a single time step of the plant, from
    initial state x and applying control signal u

    x np.array: the state of the system
    u np.array: the control signal
    """
    # do a feedforward pass on the network (model)
    x_in = np.hstack((x, u)).reshape(1, x.shape[0] + u.shape[0])
    xnext = model.predict(x_in, batch_size=1, verbose=0)
    return xnext

def compute_g_control(x, u, model, delta, idx):
    """
    Compute finite difference for each control.
    """
    u_plus = np.copy(u)
    u_minus = np.copy(u)
    u_plus[idx] = u[idx] + delta
    u_minus[idx] = u[idx] - delta
    partial_g = 1/(2*delta)*(plant_dynamics(x,u_plus, model) - plant_dynamics(x,u_minus, model))

    return partial_g

# parameters
n_steps = 200
delta = 1e-3

# load model (our system model, the neural net)
sysid = SysidModel()
sysid.load_model('./models/model')

# initialize controller
num_states = 3
num_controls = 3
controller = 0 # TODO

# define trajectory and initial conditions
x0 = np.array([0,0,0]).reshape(num_states,1)
u0 = np.zeros((num_controls,1))
xd = np.array([20,0,0]).reshape(num_states,1)

# save data
xsim = np.zeros((num_states, n_steps))
usim = np.zeros((num_controls, n_steps))

xsim[:,:1] = x0
usim[:,:1] = u0

# iterate
for i in range(1, n_steps):
    # define states and control
    x = xsim[:,i-1].reshape(num_states,1)
    u1 = usim[:,i-1].reshape(num_controls,1)

    # compute g
    g = np.zeros((num_controls,num_controls))
    for i in range(num_controls):
        # compute finite difference for each control
        g[:, i] = compute_g_control(x, u1, sysid.model, delta, i)

    # compute delta_u and u
    delta_u = np.linalg.pinv(g) @ (xd - plant_dynamics(x, u1, sysid.model).reshape(num_states,1))
    u = u1 + delta_u

    # apply u and get next states
    xsim[:,i-1:i] = plant_dynamics(x, u, sysid.model).reshape(num_states,1)
    usim[:,i-1:i] = u.reshape(num_controls,1)

# quick plots
plt.figure(0)
plt.title('states')
plt.plot(xsim[0,:])
plt.plot(xsim[1,:])
plt.plot(xsim[2,:])

plt.figure(1)
plt.title('controls')
plt.plot(usim[0,:])
plt.plot(usim[1,:])
plt.plot(usim[2,:])

plt.show()
