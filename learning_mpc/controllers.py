#!/usr/bin/env python
""" controllers.py:
Defines different controllers to control learned models.

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
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.ac = env.action_space

	def get_action(self, state):
		""" Randomly sample an action uniformly from the action space """
		return self.ac.sample()

class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in
	https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" Batches simulations through the model for speed """
		sampled_acts = np.array(
			[[self.env.action_space.sample() for j in range(
				self.num_simulated_paths)] for i in range(self.horizon)])
		states = [np.array([state] * self.num_simulated_paths)]
		nstates = []
		a = np.array([state])

		for i in range(self.horizon):
			nstates.append(self.dyn_model.predict(
				states[-1], sampled_acts[i, :]))
			
			if i < self.horizon: states.append(nstates[-1])
		costs = trajectory_cost_fn(self.cost_fn, states, sampled_acts, nstates)

		return sampled_acts[0][np.argmin(costs)]
