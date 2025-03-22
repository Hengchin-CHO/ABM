#Sample 2D functions from a Gaussian Process RBF prior and save outputs as a json file
#Charley Wu 2017

import numpy as np
import GPy
import json
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os
plt.ioff()

matplotlib.use('Agg')

def plot_kernel(imageFileName, s, xmax, observations, reverse=True):
	#plot figure
	plt.rcParams['figure.dpi'] = 300
	fig = plt.figure(figsize=(20,20))
	fig = plt.figure()
	vmin = 0
	vmax = 1
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
	font_size = {5: 10, 8: 8, 10: 7, 20: 5, 30: 2, 40: 1.5, 50: 1}
	for i in range(xmax):
		for j in range(xmax):
			if((i, j) in observations):
				plt.text(i, xmax-1-j, s_reshape[xmax-1-j, i], ha='center', fontsize=font_size[xmax])
			else:
				plt.text(i, xmax-1-j, s_reshape[xmax-1-j, i], ha='center', color='gray', fontsize=font_size[xmax])
	#save fig
	plt.savefig(imageFileName, bbox_inches='tight', format='png')
	plt.close(fig)
	plt.close()

#function to sample values from a gaussian process kernel
def samplePrior(k, imageFileName, xmin=0, xmax=10):
	"""Samples a function from a gaussian process kernel (k), where xmin and xmax specify the square bounds of the 2D function, and normalize=True scales the payoffs between 0 and 1 """
	#Create grid using xmin and xmax
	xx, yy = np.mgrid[xmin:xmax, xmin:xmax]
	X = np.vstack((xx.flatten(), yy.flatten())).T 
	K = k.K(X) #compute covariance matrix of x X
	K = np.nan_to_num(K)
	s = np.random.multivariate_normal(np.zeros(X.shape[0]), K) #GP prior distribution
	# plot_kernel('land_original.png', s, xmax, [])
	#set min-max range for scaler
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
	# #scale to range
	s_2D = min_max_scaler.fit_transform(np.reshape(s, (-1,1)))
	s = np.array([i[0] for i in s_2D])
	
	# # plot
	# plot_kernel('land_scale.png', s, xmax, [])

	#convert into JSON object
	jsonData = {}
	counter = 0
	for x1 in range(xmin,xmax):
		for x2 in range(xmin,xmax):
			jsonData[counter] = {'x1':x1, 'x2':x2, 'y':s[counter]}
			counter+=1
	return jsonData, s, K


#Create experiment data
def init_landscape(xmax, l, group_cnt):
	folder = os.path.join('landscape/images', str(xmax))
	try:
		os.mkdir(folder)
	except:
		pass
	figName = os.path.join(folder, str(group_cnt)+'.png')
	filenames = os.path.join('landscape', str(group_cnt)+'.json')

	# ************ create kernel *************
	kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=l)	# l: lambda
	(outputData, s, K) = samplePrior(kernel, figName, xmax=xmax)
	
	# ************ deal with K ***************
	K_map = {}
	for i in range(xmax):
		for j in range(xmax):
			v = K[i*xmax+j].copy()
			K_map[(i, j)] = v.reshape((xmax, -1))

	with open(filenames, 'w') as fp:
		json.dump(outputData, fp)
	
	return np.reshape(s, (xmax, xmax))