#!/usr/bin/env python
"""controller.py: 

Implement different controllers for online modeling and control.
"""

__author__ = "Jack Han-Hsun Lu"
__version__ = "0.0.1"
__date__ = "March 30, 2018"

# import
import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys

import gym

class TestController(object):
    """ Controller that takes random actions
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, state):
        """ Compute controls
        """
        control = self.action_space.sample()
        return control
        

if __name__ == '__main__':
    """ Testing controller on a simple OpenAI Gym environment: Pendulum-v1

    More info : https://gym.openai.com/envs/Pendulum-v0/

    TODO:
    Create a class to handle different plants (environments) or
    to run this loop (class to manage experiments).
    """
    # create environment (plant)
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    env.seed(0)
    n_states = env.observation_space.shape[0]
    n_controls = env.action_space.shape[0]

    # create controller
    agent = TestController(env)

    # general simulation parameters
    n_episodes = 1
    n_steps = 100
    done = False

    # store states and actions
    # TODO:
    # create a data logger class to make it cleaner
    # format: (timestep x data) x episodes
    states = np.zeros((n_steps, n_states, n_episodes))
    controls = np.zeros((n_steps, n_controls, n_episodes))

    for i in range(n_episodes):
        # get initial states
        state = env.reset()

        # log initial data
        states[0,:, i] = state
        controls[0,:, i] = np.zeros(n_controls)


        for j in range(1, n_steps):
            # compute control based on current state
            control = agent.act(state)

            # execute control and observe next states
            state, _, done, _ = env.step(control)

            # # OPTIONAL: add terminal state
            # # check if it reached some terminal state
            # # (pendulum is upright)            
            # if done:
            #     break

            # log data
            states[j,:, i] = state
            controls[j,:, i] = control

    # plot results (cycle for different episodes)
    for k in range(n_episodes):
        plt.figure()

        # plot states
        plt.subplot(2, 1, 1)
        plt.title('Episode {} out of {}'.format(k+1,n_episodes))
        for l in range(n_states):
            plt.plot(states[:,l,k], label='x{}'.format(l))

        plt.grid()
        plt.legend(loc='best')

        # plot controls
        plt.subplot(2, 1, 2)
        for m in range(n_controls):
            plt.plot(controls[:,m,k], label='u{}'.format(m))

        plt.grid()
        plt.legend(loc='best')

        # show or save plot
        plt.show()

    # close everything
    env.close()