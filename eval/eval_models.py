#!/usr/bin/env python
""" eval_models.py: 

Load models and evaluate their predictions.

# VERSION UPDATES
0.0.0 (Apr/02/2018) : initial release

"""
__author__ = "Vinicius G. Goecks"
__version__ = "0.0.0"
__date__ = "April 02, 2018"

# import
import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys
sys.path.append('../')
import time
import threading
import gym

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import SGD

from online_modeling import TestController, MemoryBuffer, ThreadingModeling

class EvalModel(object):
    """ Load and evaluate model """
    def __init__(self, env, model_id):
        self.model = load_model(model_id + '.h5')

        n_states = env.observation_space.shape[0]
        n_controls = env.action_space.shape[0]
        self.n_inputs = n_states + n_controls
        self.n_outputs = n_states


    def predict_next_states(self, current_state, control):
            """ Predict next states using current model based on current states and
            control performed.
            """
            # format input data and predict different in next states
            input_data = np.hstack((current_state, control))
            delta_next_state = self.model.predict(input_data.reshape(
                1, self.n_inputs))

            # return next states
            next_state = current_state + delta_next_state[0]
            return next_state


if __name__ == '__main__':
    """ Testing controller on a simple OpenAI Gym environment: Pendulum-v1

    More info : https://github.com/openai/gym/wiki/Pendulum-v0

    Summary:

    States [low_bound, high_bound]:
    =================================
    x0 = cos(theta) [-1.0, 1.0]
    x1 = sin(theta) [-1.0, 1.0]
    x2 = theta dot  [-8.0, 8.0]

    Initial States
    =================================
    Random angle from -pi to pi, and random velocity between -1 and 1

    Controls [low_bound, high_bound]:
    =================================
    u0 = torque [-2.0, 2.0]


    TODO:
    =================================
    Create a class to handle different plants (environments) or
    to run this loop (class to manage experiments).

    """
    PLOTTING = True

    # create environment (plant)
    run_id = 'test_eval'
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.shape[0]
    n_controls = env.action_space.shape[0]

    # create controller
    agent = TestController(env)

    # create class to evaluate model
    eval_model = EvalModel(env, model_id='../models/test1_last')

    # general simulation parameters
    n_episodes = 3
    n_steps = 500
    sim_dt = .02

    # store states (current and predicted) and actions
    # TODO:
    # create a data logger class to make it cleaner
    # format: (timestep x data) x episodes
    states = np.zeros((n_steps, n_states, n_episodes))
    pred_states = np.zeros((n_steps, n_states, n_episodes))
    controls = np.zeros((n_steps, n_controls, n_episodes))

    print('[*] Simulating...')
    for i in range(n_episodes):
        print('[*] Episode {} out of {}'.format(i+1, n_episodes))
        # get initial states
        state = env.reset()

        # log initial data
        states[0, :, i] = state
        pred_states[0, :, i] = state
        controls[0, :, i] = np.zeros(n_controls)

        for j in range(1, n_steps):
            if j%100 == 0:
                print('[*] Time step {}+/{}'.format(j,n_steps))

            # compute control based on current state
            control = agent.act(state)

            # predict next states
            pred_state = eval_model.predict_next_states(pred_states[j-1, :, i], control)

            # execute control and observe next states
            state, _, done, _ = env.step(control)

            # log data
            states[j, :, i] = state
            pred_states[j, :, i] = pred_state
            controls[j, :, i] = control


    # close everything
    env.close()

    # plot results (cycle for different episodes)
    if PLOTTING:
        # plot saved data
        for k in range(n_episodes):
            plt.figure()

            # plot states
            for l in range(n_states):
                plt.subplot(n_states+1, 1, l+1)
                if l == 0:
                    plt.title('Episode {} out of {}'.format(k+1, n_episodes))

                plt.plot(states[:, l, k], '-',
                         label='x{}'.format(l))
                plt.plot(pred_states[:, l, k], '--',
                         label='pred_x{}'.format(l))

                plt.grid()
                plt.legend(loc='best')

            # plot controls
            plt.subplot(n_states+1, 1, n_states+1)
            for m in range(n_controls):
                plt.plot(controls[:, m, k], label='u{}'.format(m))

            plt.grid()
            plt.legend(loc='best')

        # show or save plot
        plt.show()
