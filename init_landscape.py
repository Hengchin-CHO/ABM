from argparse import ArgumentParser
import GPy
import numpy as np
import random
from tqdm import tqdm
from json import JSONEncoder
import json
import os

from sklearn import preprocessing

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def samplePrior(k, xmin=0, xmax=10):
    #Create grid using xmin and xmax
    xx, yy = np.mgrid[xmin:xmax, xmin:xmax]
    X = np.vstack((xx.flatten(), yy.flatten())).T 
    K = k.K(X) #compute covariance matrix of x X
    s = np.random.multivariate_normal(np.zeros(X.shape[0]), K) #GP prior distribution
    #set min-max range for scaler
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
	# #scale to range
    s_2D = min_max_scaler.fit_transform(np.reshape(s, (-1,1)))
    s = np.array([i[0] for i in s_2D])
    return s

#add on 18/01

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def samplePriorWithRandomOffset(k1, size, xmin=0, xmax=10):
    # Create grid using xmin and xmax
    print('hellooooooooo\n', xmax//2+size//2)

    xx, yy = np.mgrid[xmin:xmax, xmin:xmax]
    X = np.vstack((xx.flatten(), yy.flatten())).T 
    
    # Compute covariance matrix for the first Gaussian Process
    K1 = k1.K(X) 
    s1 = np.random.multivariate_normal(np.zeros(X.shape[0]), K1) # GP prior distribution
    s1_2D = s1.reshape(xmax,xmax)
    # Set min-max range for scaler
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # #scale to range
    s1_2D = min_max_scaler.fit_transform(s1_2D)

    s2 = s1_2D[xmax//2-size//2:xmax//2+size//2]
    new_s2 = []
    for i in range(len(s2)):
        new_s2.append(s2[i][xmax//2-size//2:xmax//2+size//2])
    s2 = np.array(new_s2)
    s2_padded = padding(s2,xmax,xmax)
    print(s2_padded)
    s2_padded_1D = s2_padded.reshape(-1)
    # Add the padded GP distributions
    s_combined = s1 + 10*s2_padded_1D  # Combine s1 and the padded s2
    

    # Scale to range after summing
    s_2D = min_max_scaler.fit_transform(np.reshape(s_combined, (-1, 1)))  # Changed to s_combined
    s_scaled = np.array([i[0] for i in s_2D])  # Updated variable name
    
    return s_scaled  # Updated return statement

def shift_to_positive(landscape, shift_value=0.01):
    min_value = np.min(landscape)
    print(min_value)
    landscape += (np.abs(min_value) + shift_value)

    return landscape

def create_landscape(xmax, n, land_l, folder,new_method=False):
    # check the last number in this folder
    folder_contents = os.listdir(folder)
    start_i = len(folder_contents)
    print(f'- landscape number start from {start_i}')

    # create landscape and save
    kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=land_l)	# l: lambda
    for i in tqdm(range(n), ncols=70):
        if new_method:
            kernel2 = GPy.kern.RBF(input_dim=2, variance=0.2, lengthscale=land_l)	# l: lambda
            landscape = samplePriorWithRandomOffset(kernel,5,xmax=xmax)
        else:
            landscape = samplePrior(kernel, xmax=xmax)

        new_landscape = shift_to_positive(landscape)
        print(new_landscape)

        file = os.path.join(folder, f'Grid{g}_Lambda{land_l}_{str(start_i+i)}.json')
        data = {"Header": {'Grid_size': xmax, 'Lambda': land_l}, "Landscape": new_landscape}
        with open(file, "w") as f:
            json.dump(data, f, cls=NumpyArrayEncoder)
    print(f'\n- after initialize, total landscape: {start_i+n}\n')
    print('-'*40)

if __name__=='__main__':
    n = 10
    lanbdas = [3.0]
    grid_size = [40]
    new_method = False
    for land_l in lanbdas:
        for g in grid_size:
            if new_method:
                folder = os.path.join('landscape', f'Grid{g}_Lambda{land_l}_newmethod')
            else:
                folder = os.path.join('landscape', f'Grid{g}_Lambda{land_l}')
            os.makedirs(folder, exist_ok=True)
            print(f'Setting: ')
            print(f'- number of one landscape size: {n}')
            print(f'- grid size: {g}*{g}')
            print(f'- landscape lambda: {land_l}')
            print(f'- save to folder: {folder}')
            create_landscape(g, n, land_l, folder,new_method=new_method)