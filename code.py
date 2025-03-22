import numpy as np
from argparse import ArgumentParser
from agent import Agent
# from samplePrior2D import init_landscape
from create_landscape import *
from itertools import product
from record import Record
import random
from copy import deepcopy
from samplePrior2D import plot_kernel
from sklearn.preprocessing import MinMaxScaler
import os

# ********* generate one landscape, do all combination of agent and structure (loop for 9 * n) ************
parser = ArgumentParser()
parser.add_argument('-n', default=100, help='num of landscapes', type=int)
parser.add_argument('-i', default=10, help='iteration for each landscape', type=int)
parser.add_argument('-r', default=10, help='total round in one setting', type=int)
parser.add_argument('-k', default=10, help='total tiles each agent selects in one round', type=int)
parser.add_argument('-strategy', default='no_replace', help='select strategy: replace or no_replace', type=str)
parser.add_argument('-c_strategy', default='sum', help='the strategy of calculating agent\'s credibility', type=str)

def cal_credibility(actual_rewards):
    credit = {
        'sum': np.sum(actual_rewards), 
        'mean': np.mean(actual_rewards), 
        'max': np.max(actual_rewards), 
        'median': np.median(actual_rewards)
    }
    return credit

def decide_weight(agents, weights, c_strategy):
    creds = [agents[i].credit[c_strategy] for i in range(3)]
    sort_indice = np.argsort(creds)
    for i in range(len(sort_indice)):
        agents[sort_indice[i]].weight = weights[i]
        agents[sort_indice[i]].position = i


def game(landscape, agents, args, record, weights, fixed=False):
    k = args.k
    rounds = args.r
    strategy = args.strategy
    c_strategy = args.c_strategy

    gsize = record.grid_size
    t = record.freqT
    struct_type = record.struct_type
    group_number = record.group_number

    select_output_iter = record.select_output_iter
    
    # *********************** first round ***************************
    fixed_first_round = set([tuple(np.random.randint(0, gsize ,2)) for i in range(k)])
    # --- check if replace ---
    while(len(fixed_first_round) != 10):
        fixed_first_round.add(tuple(np.random.randint(0, gsize ,2)))
    # print(fixed_first_round)
    for agent in agents:
        if(fixed):
            agent.proposals = list(fixed_first_round)
            actual_rewards = [landscape[s[0], s[1]]+np.random.normal(0, 1) for s in agent.proposals]
            
            selects = {}
            for s in agent.proposals:
                selects[s] = landscape[s[0], s[1]]

            agent.update_observations(selects, first_r=True)
            agent.update_imagine()
            # --- update min_max value ---
            record.min_max[0] = min(record.min_max[0], np.amin(agent.imagine_map))
            record.min_max[1] = max(record.min_max[1], np.amax(agent.imagine_map))
        else:
            actual_rewards = []
            pos = []
            for i in range(k):
                if(i == 0):
                    agent.proposals = [tuple(np.random.randint(0, gsize ,2))]
                else:
                    agent.select_top_k(1, replace=False, first_r=True)
                reward = landscape[agent.proposals[0][0], agent.proposals[0][1]] + np.random.normal(0, 1)
                agent.update_observations({agent.proposals[0]: reward}, first_r=True)
                agent.update_imagine()
                # --- update min_max value ---
                record.min_max[0] = min(record.min_max[0], np.amin(agent.imagine_map))
                record.min_max[1] = max(record.min_max[1], np.amax(agent.imagine_map))
                # --- prepare for record ---
                actual_rewards.append(reward)
                pos.append(agent.proposals[0])
            agent.proposals = pos

        # ---- update record ----
        agent.credit = cal_credibility(actual_rewards)
        record.agent_decision(0, agent, actual_rewards)
    
        # ---- output decision map ----
        if(select_output_iter):
            path = os.path.join(map_folder, str(record.group_number))
            decision_map_output_path = os.path.join(path, 'agent'+str(agent.agent_number))
            try:
                os.mkdir(decision_map_output_path)
            except:
                pass
            file_path = os.path.join(decision_map_output_path, '0.pdf')
            observations = agent.i_observations.union(agent.g_observations)
            plot_kernel(file_path, agent.decision_map, gsize, observations)

        # ---- additional output ----
        if(agent.additional_output and select_output_iter):
            observations = agent.i_observations.union(agent.g_observations)
            record.agent_imagine_per_round[0] = deepcopy(agent.imagine_map)
            record.agent_observations_per_round[0] = observations

    # ****** decide agent structure based on structure type and weights ******
    if(struct_type == 'windfall'):
        creds = [1, 2, 3]    # dummy credit
        np.random.shuffle(creds)
    elif(struct_type == 'merit-based'):
        creds = [agent.credit for agent in agents]

    decide_weight(agents, weights, c_strategy)

    # *********************** other rounds **************************
    for r in range(1, rounds):
        group_selects = {}
        for agent in agents:
            agent.select_top_k(k, replace=False, first_r=False)
            for i in range(agent.weight):
                pos = agent.proposals[i]
                if(pos not in group_selects.keys()):
                    group_selects[pos] = landscape[pos[0], pos[1]]

            # ---- agent record ----
            actual_rewards = [landscape[s[0], s[1]]+np.random.normal(0, 1) for s in agent.proposals]
            agent.credit = cal_credibility(actual_rewards)
            record.agent_decision(r, agent, actual_rewards)

            # ---- output decision map ----
            if(select_output_iter):
                path = os.path.join(map_folder, str(record.group_number))
                decision_map_output_path = os.path.join(path, 'agent'+str(agent.agent_number))
                file_path = os.path.join(decision_map_output_path, str(r)+'.pdf')
                observations = agent.i_observations.union(agent.g_observations)
                plot_kernel(file_path, agent.decision_map, gsize, observations)

        # group record
        group_credit = cal_credibility(list(group_selects.values()))
        record.group_decision(r, group_selects, group_credit)

        # ****************** update user imagine *******************
        for agent in agents:
            agent.update_observations(group_selects, first_r=False)
            agent.update_imagine()
            # --- update min_max value ---
            record.min_max[0] = min(record.min_max[0], np.amin(agent.imagine_map))
            record.min_max[1] = max(record.min_max[1], np.amax(agent.imagine_map))
            # --- store additional output to record ---
            if(agent.additional_output and select_output_iter):
                observations = agent.i_observations.union(agent.g_observations)
                record.agent_imagine_per_round[r] = deepcopy(agent.imagine_map)
                record.agent_observations_per_round[r] = observations
        
        # ************* rotate position every t rounds  **************
        if(r % t == 0):
            decide_weight(agents, weights, c_strategy)

if __name__=='__main__':
    # test_game()
    # ********************** params setting **************************
    args = parser.parse_args()
    grid_size = [30, 40, 50]
    lambdas = [0.5, 1, 2]
    betas = [0.5, 10, 15, 20, 25, 30, 35, 40]
    # struct_types = ['windfall', 'merit-based']
    struct_types = ['merit-based']
    structures = {'IT': [2, 4, 4], 'T': [2, 2, 6], 'V': [2, 3, 5], 'P': [3, 3, 3]}
    freqT = [i for i in range(1, 11)]   # 1~10
    
    print(f'pre-defined setting: ')
    print(f'- grid size: {grid_size}')
    print(f'- lambda: {lambdas}')
    print(f'- beta: {betas}')
    print(f'- structure type: {struct_types}')
    print(f'- structure weight: {structures}')
    print(f'- credit calculate strategy: {args.c_strategy}\n')

    num_landscapes = args.n
    max_iter = args.i
    k = args.k
    print('user define setting: ')
    print(f'- number of one landscape size: {num_landscapes}')
    print(f'- iteration: {max_iter}\n')
    print(f'- total tiles each agent selects in one round: {k}\n')

    print('='*40, 'start experiment', '='*40)    
    group_cnt = 0
    for g in grid_size:
        for land_l in lambdas:
            print(f'- create landscape pool, size: {num_landscapes}')
            create_landscape(g, num_landscapes, land_l)

            for struct_type in struct_types:
                for struct_name, weights in structures.items():
                    for l in lambdas:
                        for b in list(product(*([betas]*3))):
                            for t in freqT:
                                for num in range(num_landscapes):
                                    print(f'=================== {num} =======================')
                                    land_number = num
                                    landscape = np.loadtxt(os.path.join('landscape', str(num)+'.txt'))
                                    landscape = np.reshape(landscape, (g, g))
                                    # ----- each combination run 10 times ---
                                    # select one iter output additional and decision map
                                    select_output_iter = random.randint(0, max_iter-1)
                                    print(f'- select one group output the decision map and the additional output: {select_output_iter}')
                                    
                                    
                                    for i in range(max_iter):
                                        # select one landscape number
                                        # land_number = random.randint(0, num_landscapes-1)
                                        # landscape = np.loadtxt(os.path.join('landscape', str(land_number)+'.txt'))
                                        # landscape = np.reshape(landscape, (g, g))
                                        lmin = np.amin(landscape)
                                        lmax = np.amax(landscape)

                                        # init record
                                        record = Record(group_cnt, k, is_scale=True)
                                        record.grid_size = g
                                        record.landscape_lambda = land_l
                                        record.agent_beta = b
                                        record.agent_lambda = l
                                        record.struct_type = struct_type
                                        record.structure = struct_name
                                        record.freqT = t
                                        record.min_max[0] = lmin
                                        record.min_max[1] = lmax
                                        record.land_number = land_number
                                        record.iter_number = i
                                        # only output the group equal to select_output_iter
                                        if(i == select_output_iter):
                                            record.select_output_iter = True
                                            map_folder = os.path.join('decision_map', 'map'+str(record.land_number))
                                            os.makedirs(map_folder, exist_ok=True)
                                            os.mkdir(os.path.join(map_folder, str(group_cnt)))

                                        # init agent
                                        agents = [Agent(b[j], g, l, j, False) for j in range(3)]
                                        print(f'----------- group {record.group_number} --------------')
                                        if(i == select_output_iter):
                                            rand_agent_num = random.randint(0, 2)   # 0, 1, 2
                                            agents[rand_agent_num].additional_output = True
                                            print(f'- seleced to output the expected reward: agent {rand_agent_num}')
                                        
                                        game(landscape, agents, args, record, weights)
                                        if(i == select_output_iter):
                                            record.export_additional_output()
                                        record.get_scaler()
                                        record.print_setting()
                                        record.write()

                                        group_cnt += 1