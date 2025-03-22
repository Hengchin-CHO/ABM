import numpy as np
from itertools import product
import random
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import pandas as pd
import GPy
from sklearn import preprocessing
plt.ioff()


def samplePrior(k, xmin=0, xmax=10):
    #Create grid using xmin and xmax
    xx, yy = np.mgrid[xmin:xmax, xmin:xmax]
    X = np.vstack((xx.flatten(), yy.flatten())).T 
    K = k.K(X) #compute covariance matrix of x X
    s = np.random.multivariate_normal(np.zeros(X.shape[0]), K) #GP prior distribution
    return s


def create_landscape(xmax, num, land_l):
    kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=land_l)	# l: lambda
    for i in range(num):
        filenames = os.path.join('landscape', str(i)+'.txt')
        landscape = samplePrior(kernel, xmax=xmax)
        np.savetxt(filenames, landscape)
    # s = np.loadtxt(os.path.join('landscape', str(0)+'.txt'))
