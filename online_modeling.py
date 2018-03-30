#!/usr/bin/env python
""" online_modeling.py: 

Online modeling using deep neural networks.
"""

__author__ = "Vinicius G. Goecks"
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

class MemoryBuffer(object):
    """ Stores experienced states and controls for modeling.

    Experiences are defined as:
    [current_states, control_applied] -> [(next_states - current_states)]
    
    Arguments
    ==================
    env: plant to model
    buffer_size: number of experiences to store
    """
    def __init__(self, env, buffer_size=100):
        self.env = env
        self.buffer_size = buffer_size

        # create buffer
        n_states = env.observation_space.shape[0]
        n_controls = env.action_space.shape[0]
        n_inputs = n_states + n_controls
        n_outputs = n_states

        self.buffer = np.zeros(self.buffer_size,
                               dtype=[('data_in',np.float32,(n_inputs,)),
                                      ('data_out',np.float32, (n_outputs,))])

        self.buffer_counter = 0 # counts idx of where current experience should
                                # be placed
        self.buffer_filled = False # flag becomes true when reset counter for
                                   # the first time

    def add_to_buffer(self, current_state, control, next_state):
        """ Organize data to fit buffer and manage number of experiences added.
        """
        # if full, overwrite older experiences
        if self.buffer_counter >= self.buffer_size:
            # reset buffer_counter
            self.buffer_counter = 0
            self.buffer_filled = True   

        # simplify notation
        idx = self.buffer_counter

        # add inputs (current_state, control)
        self.buffer[idx][0] = np.hstack([current_state, control])

        # add output (next_state - current_state)
        self.buffer[idx][1] = np.array([next_state - current_state])

        # increase buffer_counter
        self.buffer_counter += 1

    def generate_batch(self, batch_size=1, shuffle=False):
        """ Sample and return batch of experiences
        """
        # check if buffer is filled
        if self.buffer_filled:
            # start batch from anywhere in the buffer
            initial_idx = np.random.randint(
                low=0,
                high=(self.buffer_size - batch_size))
            final_idx = initial_idx + batch_size

        # if not, check if at least have enough experiences to return batch
        elif batch_size < self.buffer_counter:
            # start batch from anywhere in the filled part of buffer
            initial_idx = np.random.randint(
                low=0,
                high=(self.buffer_counter - batch_size))
            final_idx = initial_idx + batch_size
        else:
            # buffer doest have enough data to generate batch
            return None, None

        # sample buffer
        input_data = self.buffer['data_in'][initial_idx:final_idx]
        output_data = self.buffer['data_out'][initial_idx:final_idx]

        return input_data, output_data



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
    PLOTTING = False

    # create environment (plant)
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.shape[0]
    n_controls = env.action_space.shape[0]

    # create controller
    agent = TestController(env)

    # memory buffer
    memory = MemoryBuffer(env, buffer_size=5)

    # general simulation parameters
    n_episodes = 1
    n_steps = 10

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
            # save current state
            current_state = state

            # compute control based on current state
            control = agent.act(state)
            
            # execute control and observe next states
            state, _, done, _ = env.step(control)

            # add experience to buffer
            memory.add_to_buffer(current_state, control, state)
            input_data, output_data = memory.generate_batch(batch_size=2)

            # log data
            states[j,:, i] = state
            controls[j,:, i] = control

    # plot results (cycle for different episodes)
    if PLOTTING:
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