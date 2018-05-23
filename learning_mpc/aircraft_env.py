#!/usr/bin/env python
""" aircraft_env.py:
Defines linear aircraft model environment following OpenAI template.

VERSION CONTROL
0.0.1 (May 16, 2018): initial release
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.1"
__status__ = "Prototype"
__date__ = "May 16, 2018"

# import
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import time, os, sys
import numpy as np
from scipy import signal



class AircraftEnv(gym.Env):
    
    def __init__(self):
        """ Linear aircraft model wrapped by OpenAI Gym template."""
        # load saved model (csv format)
        model_name = 'f18a_model'
        model = np.genfromtxt(
            model_name, delimiter=',', skip_header=1)
        self.labels = np.genfromtxt(
            model_name, dtype=str, delimiter=',', max_rows=1)
        self.labels = list(self.labels)

        # organize matrices
        self.n_states = model.shape[0]
        self.n_controls = model.shape[1]-self.n_states-1 # last col is trimmed
        self.A = model[:,:self.n_states]
        self.B = model[:,self.n_states:-1]
        self.label_states = self.labels[:self.n_states]
        self.label_controls = self.labels[self.n_states:]

        # trimmed states (x0)
        self.x0 = model[:,-1].reshape(1,self.n_states)

        # adding altitude (h)
        self.n_states += 1
        self.U1 = 1004.793
        h_dot_a = np.array([[0,-self.U1,0,self.U1,0,0,0,0,0,0]])
        h_dot_b = np.array([[0,0,0]])
        # augment old a and b
        self.A = np.hstack((self.A,np.zeros((9,1))))
        self.A = np.vstack((self.A,h_dot_a))
        self.B = np.vstack((self.B,h_dot_b))

        # augment x0 and labels
        self.label_states.append('$h$ (ft)')
        h0 = 5000 # ft
        self.x0 = np.column_stack((self.x0,h0))

        # initialize C assuming full-state feedback and empty D
        self.C = np.eye(self.n_states)
        self.D = np.zeros_like(self.B)

        # create system as discretize
        self.dt = 1/50
        self.dsys = signal.cont2discrete(
            (self.A, self.B, self.C, self.D),self.dt)
        self.dA = self.dsys[0]
        self.dB = self.dsys[1]

        # ACTIONS
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.n_controls,), dtype=np.float32)

        # STATES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_states,), dtype=np.float32)

    def step(self, action):
        # PROCESS ACTION
        action = action.reshape(self.n_controls,1)
        
        # NEXT STATES
        obs = self.dA @ self.state + self.dB @ action
        self.state = np.copy(obs)

        # COMPUTE REWARD
        reward = self._compute_reward()
                
        # DONE
        # if altitude == 0 (crash)
        if obs[-1] < 0:
            done = 0
        else:
            done = 0

        # INFO
        info = {}
        self.t += 1

        return obs, reward, done, info

    def _compute_reward(self):
        return 0

    def reset(self):
        # grab initial states
        obs = self.x0.transpose()
        self.state = np.copy(obs)

        # Set time to zero.
        self.t = 0

        return obs

    def render(self, mode='human', close=False):
        # raise NotImplementedError()
        pass

    def close(self):
        pass