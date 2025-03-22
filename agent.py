import numpy as np
import GPy
from samplePrior2D import plot_kernel
import os
import copy
from sklearn import preprocessing
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def step_function(threshold, d):
    # d is distance
    if d>=threshold:
        value = 1
    else:
        value = 0
    return value
    

class Agent():
    def __init__(self, N, l, i, alpha, beta, s, replace=False):
        # ----- params -----
        self.l = l   # l: lambda
        self.beta = beta
        self.alpha = alpha
        self.position = -1
        self.agent_number = i
        self.credit = {}
        self.weight = 0
        self.replace = replace
        self.additional_output = True
        self.covariance = self.init_covariance(0, N)
        self.exploit_probability = []

        # ----- maps -----
        self.imagine_map = np.zeros((N, N), dtype=float)
        self.decision_map = np.zeros((N, N), dtype=float)
        self.uncertain_map = np.zeros((N, N), dtype=float)

        # ----- decisions -----
        self.proposals = []
        self.i_observations = set()
        self.g_observations = set()
        # visual: observations = self.i_observations.union(self.g_observations)

        # ------ sigmoid param ------
        self.s = s
        
    
    def init_covariance(self, xmin, xmax):
        kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=self.l)
        xx, yy = np.mgrid[xmin:xmax, xmin:xmax]
        X = np.vstack((xx.flatten(), yy.flatten())).T
        K = kernel.K(X)
        
        # ************ deal with K ***************
        K_map = {}
        for i in range(xmax):
            for j in range(xmax):
                v = K[i*xmax+j].copy()
                K_map[(i, j)] = v.reshape((xmax, -1))
                # print(np.round(K_map[(i, j)], 2))
        return  K_map
    
    def select_top_k(self, k, replace=False, first_r=False,choice_by_sampling=True):
        # **************** construct UCB map first *******************
        # N = self.imagine_map.shape[0]
        # UCB = np.zeros((N, N), dtype=float)
        # tmp_v = np.zeros((N, N), dtype=float)
        # for i in range(N):
        #     for j in range(N):
        #         UCB[i, j] = self.imagine_map[i, j] + self.beta * np.sqrt(self.uncertain_map[i, j])
        # # print('mx: \n', np.round(self.imagine_map, 3))
        # # print('vx: \n', np.round(self.uncertain_map, 3))
        # # print()
        # self.decision_map = UCB

        # # ************ 6/19 modified ****************
        # print(np.round(self.imagine_map, 2))
        self.exploit_probability = []
        q = self.imagine_map
        q_min = np.min(self.imagine_map)
        q_max = np.max(self.imagine_map)
        scalar = q_max - q_min

        observations = self.i_observations.union(self.g_observations)
        N = self.imagine_map.shape[0]
        m = len(observations)
        UCB = np.zeros((N, N), dtype=float)
        
        sum_m_UCB = 0
        sum_q_not_observed = 0
        q_min = np.inf
        for i in range(N):
            for j in range(N):
                if((i, j) not in observations):
                    if(q_min > q[i, j]):
                        q_min = q[i, j]
        max_di = 0
        for i in range(N):
            for j in range(N):
                if((i, j) in observations):
                    UCB[i, j] = q[i, j]
                    # print(q[i, j], end=', ')
                else:
                    #di = 0

                    di_list = []
                    for ob in observations:
                        #di += np.linalg.norm([i-ob[0], j-ob[1]])
                        di_list.append(np.linalg.norm([i-ob[0], j-ob[1]]))
                    di = min(di_list)
                    #di /= m #average distance
                    # UCB[i, j] = ((1 - self.alpha)*(q[i, j] / scalar) + self.alpha * sigmoid(di*self.s)) / (N*N - m)
                    #UCB[i, j] = (q[i, j] - q_min) + self.alpha * step_function(4, di) * np.exp(-1/(2*self.l**2))
                    #UCB[i, j] = (1 - self.alpha) * (q[i, j] - q_min) + self.alpha * step_function(7, di)
                    #UCB[i, j] = (1 - self.alpha) * (q[i, j] - q_min) + self.alpha * self.l**2 * sigmoid((di - 5) * self.s)/(5*N)
                    #UCB[i, j] = (1 - self.alpha) * (q[i, j] - q_min) / (self.l**2) + self.alpha / (N*N)
                    UCB[i, j] = (1 - self.alpha) * (q[i, j] - q_min) / (self.l**2)+ self.alpha / (N*N)
                    #UCB[i, j] = (q[i, j] - q_min) + self.alpha * sigmoid((di - 5) * self.s)
                    if di>max_di:
                        max_di = di
                    #print(f'Gaussian process value: {q[i, j] - q_min}')
                    #print(f'sigmoid value: {sigmoid((di - (np.sqrt(2 * np.square(N)) / 2)) * self.s)}')
                    sum_m_UCB += UCB[i, j]
                if (i, j) in self.i_observations and not first_r:
                    sum_m_UCB += UCB[i, j]
        for i in range(N):
            for j in range(N):
                if((i, j) not in observations):
                    sum_q_not_observed += q[i,j]-q_min
        self.exploit_probability.append(((1 - self.alpha)/(self.l**2)) *sum_q_not_observed/sum_m_UCB)
        #UCB = np.round(UCB, 2)
        #print(UCB[10].tolist())
        # print(UCB)
        # print(m)
        # print(sum_m_UCB)
        # print('----------')
        for i in range(N):
            for j in range(N):
                if((i, j) not in observations):
                    if(sum_m_UCB == 0):
                        UCB[i, j] = 0
                    else:
                        UCB[i, j] = UCB[i, j] / sum_m_UCB    
                elif (i, j) in self.i_observations:
                    UCB[i, j] = UCB[i, j] / sum_m_UCB 
        self.decision_map = UCB
        
        # print(UCB)
        # print('\n----')
        # # *********** 6/19 modified end *************
        indices = np.unravel_index(np.flip(np.argsort(UCB, axis=None)), UCB.shape)
        self.proposals = []
        if not choice_by_sampling:
            # ************************ select top k ***********************
            if(first_r and not replace):
                # exclude i_observations
                for i in range(indices[0].shape[0]):
                    if((indices[0][i], indices[1][i]) not in self.i_observations):
                        # if there are multiple position having this value
                        idx = i + 1
                        while(idx < indices[0].shape[0]):
                            if(UCB[indices[0][i], indices[1][i]] == UCB[indices[0][idx], indices[1][idx]]):
                                idx += 1
                            else:
                                break
                        random_i = np.random.randint(i, idx)
                        self.proposals = [(indices[0][random_i], indices[1][random_i])]
                        return

            elif(first_r and replace):
                i = 0
                while(idx < indices[0].shape[0]):
                    if(UCB[indices[0][i], indices[1][i]] == UCB[indices[0][idx], indices[1][idx]]):
                        idx += 1
                    else:
                        break
                random_i = np.random.randint(i, idx)
                self.proposals = [(indices[0][random_i], indices[1][random_i])]
                
            elif(not first_r):
                random_i = -1
                i = 0
                while(len(self.proposals) != k and i < indices[0].shape[0]):
                    if i == random_i:
                        i += 1
                    idx = i + 1
                    if((indices[0][i], indices[1][i]) not in self.g_observations):
                        # if there are multiple position having this value
                        while(idx < indices[0].shape[0]):
                            if(UCB[indices[0][i], indices[1][i]] == UCB[indices[0][idx], indices[1][idx]]):
                                idx += 1
                            else:
                                break
                        random_i = np.random.randint(i, idx)
                        while((indices[0][random_i], indices[1][random_i]) in self.g_observations):
                            random_i = np.random.randint(i, idx)
                        self.proposals.append((indices[0][random_i], indices[1][random_i]))
                    i += 1
                    # change for group decision because only one is chosen for alpha = 1
                    # i = idx 
                    
            # print(f'UCB: \n{np.round(UCB, 2)}')
            # print(f'proposal: {self.proposals}, len: {len(self.proposals)}')
            # print('-------------------')
        # # *********** 1/13 modified choice_by_sampling *************
        else:
            if(first_r and not replace):
                valid_indices = [(indices[0][i], indices[1][i]) for i in range(indices[0].shape[0]) 
                                 if (indices[0][i], indices[1][i]) not in self.i_observations]
                # Sample k indices from valid_indices based on UCB values
                sampled_indices = np.random.choice(len(valid_indices), k, replace=False, 
                                                    p=[UCB[valid_indices[i]] for i in range(len(valid_indices))])
                self.proposals = [valid_indices[i] for i in sampled_indices]
                
            elif(first_r and replace):
                valid_indices = [(indices[0][i], indices[1][i]) for i in range(indices[0].shape[0]) 
                                 if (indices[0][i], indices[1][i]) not in self.i_observations]
                # Sample k indices from valid_indices based on UCB values
                sampled_indices = np.random.choice(len(valid_indices), k, replace=True, 
                                                    p=[UCB[valid_indices[i]] for i in range(len(valid_indices))])
                self.proposals = [valid_indices[i] for i in sampled_indices]

            elif(not first_r):
                valid_indices = [(indices[0][i], indices[1][i]) for i in range(indices[0].shape[0]) 
                                if (indices[0][i], indices[1][i]) not in self.g_observations.union(self.i_observations)]
                # Caculate the sum of group observations
                sum_m_UCB_group = 0
                for coordinates in valid_indices:
                    i,j = coordinates[0], coordinates[1]
                    sum_m_UCB_group += UCB[i, j]
                # calculate probability based on this sum 
                UCB_group = copy.deepcopy(UCB)
                for coordinates in valid_indices:
                    i,j = coordinates[0], coordinates[1]
                    UCB_group[i, j] = UCB_group[i, j] / sum_m_UCB_group  
                # Sample k indices from valid_indices based on UCB values
                probabilities = np.array([UCB_group[valid_indices[i]] for i in range(len(valid_indices))])
                probabilities_sum = np.sum(probabilities)
                if 1 - probabilities_sum > 0.01:
                    print(f'----------- WARNING -------------\nThe sum of probabilities is equal to: {probabilities_sum}\n The sum of probabilities should be 1\n---------------------------------')
                probabilities[0] += 1 - probabilities_sum # make sure the sum of probabilities is 1, sometimes it is 0.999999
                sampled_indices = np.random.choice(len(valid_indices), k, replace=False, 
                                                    p=probabilities)
                self.proposals = [valid_indices[i] for i in sampled_indices]
        # # *********** 1/13 modified choice_by_sampling end *************
        
    def update_imagine(self):
        # update expected reward and uncertainty
        covariance = copy.deepcopy(self.covariance)
        observations = self.i_observations.union(self.g_observations)

        # print(f'agent {self.agent_number} observation: {observations}')

        y = [self.imagine_map[i[0], i[1]] for i in observations]
        K = []  # 大K
        sigma = 0.1
        for s1 in observations:
            K.append([covariance[s1][s2[0], s2[1]] for s2 in observations])
        K_inv = np.linalg.inv(np.array(K) + np.identity(len(K))*np.square(sigma))  # inverse of (K + I)

        # for x in all other tiles
        gmax = self.imagine_map.shape[0]
        for i in range(gmax):
            for j in range(gmax):
                k = np.array([covariance[(i, j)][s[0], s[1]] for s in observations])   # 小k
                mx = np.matmul(np.matmul(k.T,  K_inv), y)
                vx = covariance[(i, j)][i, j] - np.matmul(np.matmul(k.T,  K_inv),  k)
                if((i, j) not in observations):
                    self.imagine_map[i, j] = mx
                    self.uncertain_map[i, j] = vx
                else:
                    self.uncertain_map[i, j] = 0
    
    def update_observations(self, selects, first_r):
        # selects: {(x, y): r, ...}, K = {(x, y): [[c00, c01, ...], [c10, c11, ...]], }
        for i, yt in selects.items():
            self.imagine_map[i[0], i[1]] = yt
            if(first_r):
                self.i_observations.add(i)
            else:
                self.g_observations.add(i)
