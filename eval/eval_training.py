#!/usr/bin/env python
""" eval_training.py: 

Load training results, average, and plot them.

# VERSION UPDATES
0.0.0 (Apr/02/2018) : initial release

"""
__author__ = "Vinicius G. Goecks"
__version__ = "0.0.0"
__date__ = "April 02, 2018"

# import
import numpy as np
import matplotlib.pyplot as plt

# load data
data_addr = '../models/'
runs = ['test', 'test1','test2']

# plot them
plt.figure()
for run in runs:
    raw_data = np.load(data_addr + run + '_hist.npy')
    val_loss = raw_data[:,1]
    plt.plot(val_loss,label=str(run))

plt.grid()
# plt.xlim([0,500])
plt.legend(loc='best')
plt.show()