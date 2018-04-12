#!/usr/bin/env python
"""viz_data.py: visualize data acquired.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "February 08, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt("../data/data_out_test.csv", delimiter=",")
in_out_type = True

# plot each state
# load stylesheet
try:
    plt.style.use("dwplot")
except:
    print("Cannot use this stylesheet")

if in_out_type:
    for i in range(0,data.shape[1]):
        plt.figure(i)
        plt.title('Reading %i' %i)
        plt.plot(data[:,i])
        plt.grid()

else:
    for i in range(1,data.shape[1]):
        plt.figure(i)
        plt.title('Reading %i' %i)
        plt.plot(data[:,0],data[:,i])
        plt.grid()

plt.show()
