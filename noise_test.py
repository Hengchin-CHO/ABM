import numpy as np
from argparse import ArgumentParser
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import GPy

# ********* generate one landscape, do all combination of agent and structure (loop for 9 * n) ************
parser = ArgumentParser()
parser.add_argument('-noise_strategy', default='normal', help='the strategy of noise range (normal or 10_percent)', type=str)

def samplePrior(k, xmin=0, xmax=10):
    #Create grid using xmin and xmax
    xx, yy = np.mgrid[xmin:xmax, xmin:xmax]
    X = np.vstack((xx.flatten(), yy.flatten())).T 
    K = k.K(X) #compute covariance matrix of x X
    K = np.nan_to_num(K)
    s = np.random.multivariate_normal(np.zeros(X.shape[0]), K) #GP prior distribution
    return s

def create_landscape(xmax, num, land_l):
    kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=land_l)	# l: lambda
    for i in range(num):
        filenames = os.path.join('landscape', str(i)+'.txt')
        landscape = samplePrior(kernel, xmax=xmax)
        np.savetxt(filenames, landscape)

def land_add_noise(landscape, noise_strategy):
    if(noise_strategy == 'normal'):
        noise = np.random.normal(0, 1, size=landscape.shape)
    elif(noise_strategy == '10_percent'):
        noise = np.zeros(landscape.shape)
        for i in range(landscape.shape[0]):
            for j in range(landscape.shape[1]):
                r = landscape[i][j]
                noise[i][j] = np.random.uniform(-0.1*r, 0.1*r)
    noise_landscape = np.add(landscape, noise)

    return noise_landscape

def plot_kernel(imageFileName, s, xmax, vmin, vmax, reverse=True):
	fig = plt.figure()
	# vmin = np.amin(s)
	# vmax = np.amax(s)

	if(reverse):
		s = np.flip(s.T, 0)	# (0, 0) on left-down
	plt.matshow(s.reshape((xmax, xmax)), interpolation='none', cmap=plt.get_cmap('Spectral_r'), vmin=vmin, vmax=vmax)
	plt.colorbar(label='m(x)', shrink=0.75)
	plt.xticks(np.arange(0.5,xmax+0.5,1), [])
	plt.yticks(np.arange(0.5, xmax+.5,1), [])
	plt.tick_params(axis='both', which='both',length=0)
	plt.grid(color='#0082C1', linestyle='-', linewidth=0.5)

	s_reshape = np.round(s.reshape((xmax, xmax)), 2)
	font_size = 2
	for i in range(xmax):
		for j in range(xmax):
			plt.text(i, xmax-1-j, s_reshape[xmax-1-j, i], ha='center', color='gray', fontsize=font_size)
	plt.savefig(imageFileName, bbox_inches='tight', format='pdf')
	plt.close(fig)
	plt.close()

def plot_distribution(imageFileName, landscape, noise_landscape):
    landscape = landscape.reshape((1, -1))[0]
    noise_landscape = noise_landscape.reshape((1, -1))[0]

    fig = plt.figure()
    plt.hist(landscape,bins=30,density=True,alpha=0.3, histtype='stepfilled',color='r', edgecolor='none', label='original')
    plt.hist(noise_landscape,bins=30,density=True,alpha=0.3, histtype='stepfilled',color='b', edgecolor='none', label='noise')
    plt.legend()

    #save fig
    plt.savefig(imageFileName, bbox_inches='tight', format='pdf')
    plt.close(fig)
    plt.close()

if __name__=='__main__':
    # ********************** params setting **************************
    args = parser.parse_args()
    noise_strategy = args.noise_strategy
    num_landscapes = 2
    g = 10
    land_l = 0.05
    land_num = []
    all_lmax = []
    all_lmin = []
    create_landscape(g, num_landscapes, land_l)
    os.mkdir('noise_test_output')
    for num in range(num_landscapes):
        landscape = np.loadtxt(os.path.join('landscape', str(num)+'.txt'))
        landscape = np.reshape(landscape, (g, g))

        noise_landscape = land_add_noise(landscape, noise_strategy)
        difference_landscape = np.subtract(noise_landscape, landscape)

        lmin = min(np.amin(noise_landscape), np.amin(landscape), np.amin(difference_landscape))
        lmax = max(np.amax(noise_landscape), np.amax(landscape), np.amax(difference_landscape))
        
        file_name = os.path.join('noise_test_output', str(num) + '.pdf')
        plot_kernel(file_name, landscape, g, lmin, lmax, reverse=True)
        
        file_name = os.path.join('noise_test_output', str(num) + '_noise.pdf')
        plot_kernel(file_name, noise_landscape, g, lmin, lmax, reverse=True)

        file_name = os.path.join('noise_test_output', str(num) + '_difference.pdf')
        plot_kernel(file_name, difference_landscape, g, lmin, lmax, reverse=True)

        file_name = os.path.join('noise_test_output', str(num) + '_distribution.pdf')
        plot_distribution(file_name, landscape, noise_landscape)

        land_num.append(num)
        all_lmax.append(np.amax(landscape))
        all_lmin.append(np.amin(landscape))

    record_min_max = pd.DataFrame(columns=['land_num', 'min', 'max'])
    record_min_max['land_num'] = land_num
    record_min_max['min'] = all_lmin
    record_min_max['max'] = all_lmax
    file_name = os.path.join('noise_test_output', 'min_max.csv')
    record_min_max.to_csv(file_name, index=False)