#!/usr/bin/env python
""" cost_function.py:
Defines cost function for MPC controller.

Based on material presented at Berkeley's Deep Reinforcement Learning
course (Fall 2017):  
http://rll.berkeley.edu/deeprlcourse/

VERSION CONTROL
0.0.1 (May 23, 2018): initial release
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.1"
__status__ = "Prototype"
__date__ = "May 23, 2018"

# import
import numpy as np
import sys

#========================================================
# 
# Environment-specific cost functions:
#

def pendulum_cost_fn(state, action, next_state):
    # compute costs with respect to goal state
    # states: np.array([np.cos(theta), np.sin(theta), thetadot])

    # need to check len(state.shape) because sometimes we get just one path,
    # instead of many
    goal_state = np.array([-1,0,0])
    if len(state.shape) > 1:
        scores = np.sum(np.sqrt((goal_state-state)**2), axis=1)
    else:
        scores = np.sum(np.sqrt((goal_state-state)**2), axis=0)
    return scores


def aircraft_cost_fn(state, action, next_state):
    # compute costs with respect to goal state
    # TODO
    pass

#========================================================
# 
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], next_states[i])

    return trajectory_cost