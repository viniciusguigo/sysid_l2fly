#!/usr/bin/env python
"""main.py: implement deep rl for sysid and control.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "March 09, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import LSTM

from predict_gtm import SysidModel
from control import Neural_iLQR

# parameters
n_iter = 1 # number of iterations for appending and updating model, after initial
best_cost = 1e9

# load model (our system model, the neural net)
sysid = SysidModel()
sysid.load_model('model')

# initialize controller
num_states = 4
num_controls = 2
controller = Neural_iLQR(model=sysid.model,n_x=num_states,n_u=num_controls)

# generate initial trajectory
print('-----\nIteration 0')
x0 = np.array([0.3,0.3,0.3,0.3]).reshape(1,4)
U0 = np.zeros((40,2))
Xf, Uf, cost_f = controller.ilqr(x0,U0)

# check for best cost
if cost_f < best_cost:
    print('* Found best model. Cost = ', cost_f)
    best_cost = cost_f
    sysid.save_model(controller.model,'best_model')


# iterate
for i in range(1,n_iter):
    print('-----\nIteration ',i)
    # append data and fit model
    new_model = sysid.append_and_fit(controller.model, Xf, Uf)
    controller.model = new_model

    # generate more trajectories
    Xf, Uf, cost_f = controller.ilqr(x0,U0)

    # check for best cost
    if cost_f < best_cost:
        print('* Found best model. Cost = ', cost_f)
        best_cost = cost_f
        sysid.save_model(new_model,'best_model')

# use best model
print('-----\nFinal Iteration - Loading best model.')
controller.model = sysid.load_model('best_model')
Xf, Uf, cost_f = controller.ilqr(x0,U0)
controller.plot_neural_ilqr(Xf,Uf)

# compare again with original learned model and original data
sysid.compare_models(sysid.load_model('model'),controller.model)
