import numpy as np
from argparse import ArgumentParser
from agent import Agent
# from samplePrior2D import init_landscape
# from create_landscape import *
from itertools import product
from itertools import combinations_with_replacement
from record import Record
import random
from copy import deepcopy
from samplePrior2D import plot_kernel
from sklearn.preprocessing import MinMaxScaler
import os

import threading
from tqdm import tqdm
from json import JSONEncoder
import json
import os

from copy import deepcopy

from time import time

#np.random.seed(363)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# ********* generate one landscape, do all combination of agent and structure (loop for 9 * n) ************
parser = ArgumentParser()
parser.add_argument('-n', default=1, help='num of landscapes', type=int)
parser.add_argument('-i', default=30, help='iteration for each landscape', type=int)
parser.add_argument('-r', default=20, help='total round in one setting', type=int)
parser.add_argument('-k', default=5, help='total tiles each agent selects in one round', type=int)
parser.add_argument('-strategy', default='no_replace', help='select strategy: replace or no_replace', type=str)
parser.add_argument('-c_strategy', default='mean', help='the strategy of calculating agent\'s credibility', type=str)
parser.add_argument('-noise_strategy', default='normal', help='the strategy of noise range (normal or 10_percent)', type=str)
parser.add_argument('-fixed_leader', default=[True, [0.7,0.2,0.1]], help='Is the leader always the same. If yes, the first alpha is the leader.', type=list)
parser.add_argument('-fixed_start', default='worst', help='Fix first round observs. "not_fixed","best", "worst" or "manual" are available.', type=str)
parser.add_argument('-freqT', default=[4], help='Turnover frequency', type=list)


parser.add_argument('-s', default=1, help='sigmoid', type=float)
parser.add_argument('-fr_value', default=1, help='number of first round', type=int)


def cal_credibility(actual_rewards):
    credit = {
        'sum': np.sum(actual_rewards), 
        'mean': np.mean(actual_rewards), 
        'max': np.max(actual_rewards), 
        'median': np.median(actual_rewards)
    }
    return credit

def reweight(weights, k):
    base_counts = [int(weight * k) for weight in weights]
    
    remainder = k - sum(base_counts)
    
    weighted_indices = np.argsort(weights)[::-1]
    for i in range(remainder):
        base_counts[weighted_indices[i]] += 1
    
    return base_counts

def decide_weight(agents, weights, creds, k,turnover=False):
    if turnover:
        new_weights = weights
    else:
        # 0908 reweight ------------
        new_weights = reweight(weights, k)
        # print(f'weight: {weights}, new weight: {new_weights}')

    sort_indice = np.argsort(creds)[::-1]
    for i in range(len(sort_indice)):
        agents[sort_indice[i]].weight = new_weights[i]
        agents[sort_indice[i]].position = i



# fr = first round
def sum_of_max_in_first_round(agent, record, fr_value, k, add_noise):
    sum_of_max = 0
    for v in range(1, fr_value):
        agent.select_top_k(k, replace=False, first_r=False)
        actual_rewards = [landscape[s[0], s[1]] for s in agent.proposals]
        noise = [0 for _ in range(len(agent.proposals))]
        if(add_noise):
            noise = [np.random.normal(0, 1) for _ in range(len(agent.proposals))]
        actual_rewards = [actual_rewards[i] + noise[i] for i in range(len(actual_rewards))]
        
        selects = {}
        for i in range(len(agent.proposals)):
            selects[agent.proposals[i]] = actual_rewards[i]

        agent.update_observations(selects, first_r=True)
        agent.update_imagine()

        # update maximum value
        sum_of_max += np.max(actual_rewards)
        
        # update record
        record.agent_decision(0, agent, actual_rewards)

    return sum_of_max

def select_landscape_in_pre_phase(folder, N, land_number):
    folder_contents = os.listdir(folder)
    if(len(folder_contents) < N+1):
        print('\nError! there isn\'t enough landscapes! pre phase')
        exit()
    lands = list(np.arange(0, len(folder_contents)))
    if(int(land_number) in lands):
        lands.remove(int(land_number))

    pre_phase_land_idx = np.random.choice(lands, size=N, replace=False)
    
    return pre_phase_land_idx


def pre_phase(agents_setting, N, weights, folder, land_number, k, setting, gsize, record):
    '''
    要在 prephase 做到的事情:
    1. 抽選 N 個 landscape (不包含即將跑的那個)
    2. 計算每次的結果 (max) 並記錄
    3. 根據結果決定 credit
    4. 設定好 agent 的權重
    5. print & return
    '''
    # ------------- select N landscapes ---------------
    #pre_phase_land_idx = select_landscape_in_pre_phase(folder, N, land_number)
    pre_phase_land_idx = [1]

    # ------------- agent select and record ------------
    sum_of_max_per_agent = [0, 0, 0]
    for n in range(N):
        agents = [Agent(*agents_setting[j]) for j in range(3)]
        # read landscape
        land_number = pre_phase_land_idx[n]
        file = os.path.join('landscape', setting, f'{setting}_{str(land_number)}.json')
        landscape = read_landscape(file, gsize)
        for a in range(len(agents)):
            agent = agents[a]
            # ------------- select one-by-one ---------------
            actual_rewards = []
            pos = []
            exploit_prob = []
            for i in range(k):
                if(i == 0):
                    agent.proposals = [tuple(np.random.randint(0, gsize ,2))]
                else:
                    agent.select_top_k(1, replace=False, first_r=True)
                    exploit_prob.append(agent.exploit_probability[0])

                reward = landscape[agent.proposals[0][0], agent.proposals[0][1]]
                
                agent.update_observations(deepcopy({agent.proposals[0]: reward}), first_r=True)
                agent.update_imagine()
                
                actual_rewards.append(reward)
                pos.append(agent.proposals[0])
                

            agent.proposals = pos
            agent.exploit_probability = exploit_prob
            # update maximum value
            sum_of_max_per_agent[a] += np.max(actual_rewards)
            
            # update record
            agent.credit = cal_credibility(actual_rewards)
            agent.credit_observation = cal_credibility(actual_rewards)

            record.agent_decision(-N+n, agent, actual_rewards)
        record.update_group_mapping(f'{setting}_{str(land_number)}', n-N)
        
    # ------------ decide credit ------------
    creds = [sum_of_max_per_agent[i] / N for i in range(len(sum_of_max_per_agent))]
    # decide_weight(agents, weights, creds, k)

    # ----------- print -------------
    print(f'- Pre-phase landscape: {pre_phase_land_idx}')

    return creds


def game(landscape, agents, args, record, weights, fixed=False, add_noise=False, group_sampling=True, creds = None):
    if args.fixed_leader[0]:
        fixed_chief = True
        fixed_chief_weights = args.fixed_leader[1]
    else:
        fixed_chief = False
    k = args.k
    rounds = args.r
    strategy = args.strategy
    c_strategy = args.c_strategy
    noise_strategy = args.noise_strategy

    gsize = record.grid_size
    t = record.freqT
    struct_type = record.struct_type
    group_number = record.group_number

    select_output_iter = record.select_output_iter

    map_folder = os.path.join('decision_map', str(record.land_number), str(record.group_number))
    # if not prephase only
    if rounds!= -1:
        # *********************** first round ***************************
        #fixed_first_round = set([tuple(np.random.randint(0, gsize ,2)) for i in range(k)])
        if fixed:
            indices_for_fixed = np.unravel_index(np.flip(np.argsort(landscape, axis=None)), landscape.shape)
            #best ones
            if args.fixed_start == 'best':
                fixed_first_round = [(indices_for_fixed[0][i], indices_for_fixed[1][i]) for i in range(k)]
            elif args.fixed_start == 'worst':
                fixed_first_round = [(indices_for_fixed[0][-i-1], indices_for_fixed[1][-i-1]) for i in range(k)]
            elif args.fixed_start == 'manual':
                fixed_first_round = [(0, 0), (3, 3)]
            # --- check if number of observations is correct ---
            while(len(fixed_first_round) != k):
                fixed_first_round.append(tuple(np.random.randint(0, gsize ,2)))
                print('fixed_first_round does not contain enough observations. Add one more.')
        # print(fixed_first_round)
        agent_id = 0
        for agent in agents:
            # ----- create decision map folder ----
            if(select_output_iter and record.is_output_decision_map):
                path = os.path.join(map_folder, 'agent'+str(agent.agent_number))
                os.mkdir(path)
            # ---- select ----
            if(fixed):
                agent.proposals = list(fixed_first_round)
                actual_rewards = [landscape[s[0], s[1]] for s in agent.proposals]
                noise = [0 for _ in range(len(agent.proposals))]
                if(add_noise):
                    noise = [np.random.normal(0, 1) for _ in range(len(agent.proposals))]
                actual_rewards = [actual_rewards[i] + noise[i] for i in range(len(actual_rewards))]
                
                selects = {}
                for i in range(len(agent.proposals)):
                    selects[agent.proposals[i]] = actual_rewards[i]

                agent.update_observations(selects, first_r=True)
                agent.update_imagine()
                # --- update min_max value ---
                record.min_max[0] = min(record.min_max[0], np.amin(agent.imagine_map))
                record.min_max[1] = max(record.min_max[1], np.amax(agent.imagine_map))
                exploit_prob = [0 for _ in range(len(agent.proposals))]
                agent.exploit_probability = exploit_prob
            else:
                actual_rewards = []
                pos = []
                exploit_prob = []
                for i in range(k):
                    if(i == 0):
                        agent.proposals = [tuple(np.random.randint(0, gsize ,2))]
                    else:
                        agent.select_top_k(1, replace=False, first_r=True)
                        exploit_prob.append(agent.exploit_probability[0])

                    reward = landscape[agent.proposals[0][0], agent.proposals[0][1]]
                    noise = 0
                    if(add_noise):
                        if(noise_strategy == 'normal'):
                            noise = np.random.normal(0, 1)
                        elif(noise_strategy == '10_percent'):
                            noise = np.random.uniform(-0.1*reward, 0.1*reward)
                    reward = reward + noise

                    #if(i == k-1):
                    # ---- output decision map ----
                    if(select_output_iter and record.is_output_decision_map):
                        path = os.path.join(map_folder, 'agent'+str(agent.agent_number))
                        file_path = os.path.join(path, f'individual_{i}.png')
                        observations = agent.i_observations.union(agent.g_observations)
                        plot_kernel(file_path, agent.decision_map, gsize, observations)

                        file_path = os.path.join(path, f'individual_{i}.json')
                        data = {"Landscape": np.round(agent.decision_map, 2).reshape((-1))}
                        with open(file_path, "w") as f:
                            json.dump(data, f, cls=NumpyArrayEncoder)
                    # ---- output first round decision map if agent lambda == land lambda ----
                    #if(select_output_iter and record.is_output_first_decision_map and agent.l == record.landscape_lambda):
                    if(select_output_iter and record.is_output_first_decision_map):
                        folder = os.path.join('decision_map', 'first_round', str(record.land_number))
                        file_path = os.path.join(folder, str(record.group_number), f'agent{str(agent.agent_number)}_{i}tile.png')
                        #file_path = os.path.join(folder, str(record.group_number), f'agent{str(agent.agent_number)}.png')
                        observations = agent.i_observations.union(agent.g_observations)
                        plot_kernel(file_path, agent.decision_map, gsize, observations)
                        file_path = os.path.join(folder, str(record.group_number), f'agent{str(agent.agent_number)}_{i}tile.json')
                        #file_path = os.path.join(folder, str(record.group_number), f'agent{str(agent.agent_number)}.json')
                        data = {"Landscape": np.round(agent.decision_map, 2).reshape((-1))}
                        with open(file_path, "w") as f:
                            json.dump(data, f, cls=NumpyArrayEncoder)

                    agent.update_observations({agent.proposals[0]: reward}, first_r=True)
                    agent.update_imagine()
                    # --- update min_max value ---
                    record.min_max[0] = min(record.min_max[0], np.amin(agent.imagine_map))
                    record.min_max[1] = max(record.min_max[1], np.amax(agent.imagine_map))
                    # --- prepare for record ---
                    actual_rewards.append(reward)
                    pos.append(agent.proposals[0])
                agent.proposals = pos
                agent.exploit_probability = exploit_prob

            # ---- update record ----
            agent.credit = cal_credibility(actual_rewards)
            agent.credit_observation = cal_credibility(actual_rewards)
            record.agent_decision(0, agent, actual_rewards)
            record.min_max[0] = min(record.min_max[0], np.amin(actual_rewards))
            record.min_max[1] = max(record.min_max[1], np.amax(actual_rewards))

            # ---- additional output ----
            if(agent.additional_output and select_output_iter):
                observations = agent.i_observations.union(agent.g_observations)
                record.agent_imagine_per_round[0] = deepcopy(agent.imagine_map)
                record.agent_observations_per_round[0] = observations

            # ------- 8/29 updated -------
            # fr_value = args.fr_value
            # sum_of_max = np.max(actual_rewards)
            
            # sum_of_max += sum_of_max_in_first_round(agent, record, fr_value, k, add_noise)
            # agent.credit[c_strategy] = sum_of_max / fr_value

        # ****** decide agent structure based on structure type and weights ******
        # if(struct_type == 'windfall'):
        #     creds = [1, 2, 3]    # dummy credit
        #     np.random.shuffle(creds)
        # elif(struct_type == 'merit-based'):
        #     creds = [agent.credit[c_strategy] for agent in agents]
        # decide_weight(agents, weights, creds, k)

    # *********************** other rounds **************************
    # If group-sampling, then the agent's weight is based on the initial hierarchy structure without modification.
    if group_sampling:
            if fixed_chief:
                for i in range(len(agents)):
                    agents[i].weight = fixed_chief_weights[i]
                    agents[i].position = i
            elif creds is not None:
                sort_indice = np.argsort(creds)
                for i in range(len(sort_indice)):
                    agents[sort_indice[i]].weight = weights[i]/np.sum(weights)
                    agents[sort_indice[i]].position = i
            else: 
                raise ValueError('group sampling should have creds')
    for agent in agents:
        agent.observation_values_for_turnover = []
    for r in range(1, rounds):
        group_selects = {}
        for agent in agents:
            agent.observation_values = []
            agent.select_top_k(k, replace=False, first_r=False)
            noise = [0 for _ in range(len(agent.proposals))]
            if(add_noise):
                if(noise_strategy == 'normal'):
                    noise = [np.random.normal(0, 1) for _ in range(len(agent.proposals))]
                elif(noise_strategy == '10_percent'):
                    noise = [np.random.uniform(-0.1*landscape[pos[0], pos[1]], 0.1*landscape[pos[0], pos[1]]) for pos in agent.proposals]

            i, j = 0, 0
            if not group_sampling:
                while(i < agent.weight and j < len(agent.proposals)):
                    # print(agent.weight, agent.proposals)
                    pos = agent.proposals[j]
                    agent.observation_values.append(landscape[pos[0], pos[1]])
                    agent.observation_values_for_turnover.append(landscape[pos[0], pos[1]])
                    if(pos not in group_selects.keys()):
                        group_selects[pos] = landscape[pos[0], pos[1]] + noise[j]
                    i += 1
                    j += 1
        
            # ---- output decision map ----
            if(select_output_iter and record.is_output_decision_map):
                decision_map_output_path = os.path.join(map_folder, 'agent'+str(agent.agent_number))
                file_path = os.path.join(decision_map_output_path, f'round{r}.png')
                observations = agent.i_observations.union(agent.g_observations)
                plot_kernel(file_path, agent.decision_map, gsize, observations)

                file_path = os.path.join(decision_map_output_path,f'round{r}.json')
                data = {"Landscape": np.round(agent.decision_map, 2).reshape((-1))}
                with open(file_path, "w") as f:
                    json.dump(data, f, cls=NumpyArrayEncoder)
            # update min_max value
            record.min_max[0] = min(record.min_max[0], np.amin(actual_rewards))
            record.min_max[1] = max(record.min_max[1], np.amax(actual_rewards))

            
        
        if group_sampling:
            probabilities_group_sampling = [agent.weight for agent in agents]
            agent_current_idx = [0 for agent in agents]
            for i in range(k):
                chosen_agent = np.random.choice(len(agents), p=probabilities_group_sampling)
                agent = agents[chosen_agent]
                # we chose the best imagine position not yet given to the group
                pos = agent.proposals[agent_current_idx[chosen_agent]]
                agent.observation_values.append(landscape[pos[0], pos[1]])
                agent.observation_values_for_turnover.append(landscape[pos[0], pos[1]])
                agent_current_idx[chosen_agent] += 1
                #print(f'Round {r}: agent {agent.agent_number} at position {agent.position} select {pos} with weight {agent.weight} as tile number {i}')
                if(pos not in group_selects.keys()):
                    if add_noise:
                        group_selects[pos] = landscape[pos[0], pos[1]] + noise[chosen_agent]
                    else:  
                        group_selects[pos] = landscape[pos[0], pos[1]]
        
        # group record
        # print(group_selects, '\n')
        for agent in agents:
            # ---- agent record ----
            actual_rewards = [landscape[s[0], s[1]] for s in agent.proposals]
            actual_rewards = [actual_rewards[i] + noise[i] for i in range(len(actual_rewards))]
            agent.credit = cal_credibility(actual_rewards)
            if len(agent.observation_values)==0:
                agent.credit_observation = cal_credibility([-1])
            else:
                agent.credit_observation = cal_credibility(agent.observation_values)
            record.agent_decision(r, agent, actual_rewards,c_strategy=c_strategy)


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
            for agent in agents:
                if len(agent.observation_values_for_turnover)==0:
                    agent.observation_values_for_turnover = [-1]
            creds = [cal_credibility(agent.observation_values_for_turnover)[c_strategy] for agent in agents]
            if args.fixed_leader[0]:
                decide_weight(agents, args.fixed_leader[1], creds,k,turnover=True)
                print('We have a fixed leader, but it changes after the first turnover.')
            else:
                decide_weight(agents, weights, creds,k,turnover=True)
            for agent in agents:
                agent.observation_values_for_turnover = []
            #from proposals even agents didn't observe
            #creds = [agent.credit[c_strategy] for agent in agents]
            #decide_weight(agents, weights, creds,k,turnover=True)

        


def one_game_job(args, record, weights, landscape, setting):
    beta = record.agent_beta
    alpha = record.agent_alpha
    g = record.grid_size
    l = record.agent_lambda

    s = record.s
    # init agent
    agents = [Agent(g, l, j, alpha[j], beta[j], s, False) for j in range(3)]
    # print(f'----------- group {record.group_number} --------------')
    if(record.select_output_iter and record.is_output_additional_output):
        rand_agent_num = random.randint(0, 2)   # 0, 1, 2
        agents[rand_agent_num].additional_output = True
        print(f'- selected to output the expected reward: agent {rand_agent_num}')

    if args.fixed_start == 'not_fixed':
        fixed = False
    else:
        fixed = True
    add_noise = False

    # 0908: pre-phase -------------------------------
    N = args.fr_value
    land_number = record.land_number.split('_')[-1]
    folder = os.path.join('landscape', setting)
    agents_setting = [(g, l, j, alpha[j], beta[j], s, False) for j in range(3)]
    creds = pre_phase(agents_setting, N, weights, folder, land_number, k, setting, g, record)
    
    # -----------------------------------------------
    agents = [Agent(g, l, j, alpha[j], beta[j], s, False) for j in range(3)]
    decide_weight(agents, weights, creds, k)

    game(landscape, agents, args, record, weights, fixed, add_noise,creds=creds)
    for round in range(args.r):
        record.update_group_mapping(f'{setting}_{str(land_number)}', round)
    record.get_scaler()

def select_landscape(g, n, land_l):
    folder = os.path.join('landscape', f'Grid{g}_Lambda{land_l}')
    folder_contents = os.listdir(folder)
    if(len(folder_contents) < n):
        print('\nError! there isn\'t enough landscapes!')
        exit()
    random_land_idx = np.random.choice(np.arange(0, len(folder_contents)), size=n, replace=False)
    return random_land_idx

def read_landscape(file, g):
    with open(file, "r") as f:
        json_f = json.load(f)
        landscape = np.reshape(np.asarray(json_f["Landscape"]), (g, g))
    return landscape



if __name__=='__main__':
    os.makedirs('decision_map', exist_ok=True)
    # ********************** params setting **************************
    args = parser.parse_args()
    grid_size = [40]
    land_lambdas = [3.0]
    agent_lambdas = [1]
    # betas = [0.5, 10, 15, 20, 25, 30, 35, 40]
    betas = [1, 1, 1]
    #alpha = np.linspace(0,1,9)
    alpha = [0.01,0.99]
    alpha_for_chief = [(0.01,0.01,0.99),(0.99,0.99,0.01),(0.01,0.99,0.99),(0.99,0.01,0.01)]#(0.01,0.01,0.01),(0.99,0.99,0.99)
    #alpha = [0, 0.5,1]

    struct_types = ['merit-based']
    #struct_types = ['windfall', 'merit-based']
    structures = {'V': [0.1, 0.2, 0.7]}#'V': [0.1, 0.2, 0.7]
    #structures = {'IT': [0.2, 0.4, 0.4], 'T': [0.2, 0.2, 0.6], 'V': [0.2, 0.3, 0.5], 'P': [0.3, 0.3, 0.3]}


    freqT = args.freqT
    is_output_decision_map = False
    is_output_first_decision_map = False
    is_output_additional_output = False
    
    print(f'pre-defined setting: ')
    print(f'- grid size: {grid_size}')
    print(f'- landscape lambda: {land_lambdas}')
    print(f'- agent lambda: {agent_lambdas}')
    # print(f'- beta: {betas}')
    print(f'- alpha: {alpha}')
    print(f'- structure type: {struct_types}')
    print(f'- structure weight: {structures}')
    print(f'- credit calculate strategy: {args.c_strategy}\n')

    n = args.n
    max_iter = args.i
    k = args.k
    s = args.s
    fr_value = args.fr_value
    print('user define setting: ')
    print(f'- random select {n} landscapes from one folder')
    print(f'- iteration: {max_iter}')
    print(f'- total tiles each agent selects in one round: {k}')
    print(f'- sigmoid: {s}')
    print(f'- number of first round: {fr_value}\n')
    print('='*40, 'start experiment', '='*40)    
    group_cnt = 0
    starttime = time()
    for g in grid_size:
        for land_l in land_lambdas:
            for l in agent_lambdas:
                for t in freqT:
                    #alpha_to_go_through = [tuple(alpha[i:i+3]) for i in range(0,len(alpha),3)]
                    #for a in alpha_to_go_through:
                    for a in combinations_with_replacement(alpha, 3):
                    #for a in alpha_for_chief:
                        for b in list([betas]):
                            for struct_type in struct_types:
                                for struct_name, weights in structures.items():
                                    print(f'--------------- current setting --------------')
                                    print(f'- grid size: {g}*{g}')
                                    print(f'- landscape lambda: {land_l}')
                                    # print(f'- agent beta: {b}')
                                    print(f'- agent alpha: {a}')
                                    print(f'- agent lambda: {l}')
                                    print(f'- structure: {struct_name} ({struct_type})')
                                    print(f'- freqT: {t}')
                                    print(f'- group number: {group_cnt} ~ {group_cnt + n*max_iter}')
                                    #random_land_idx = select_landscape(g, n, land_l)
                                    random_land_idx = [1]
                                    setting = f'Grid{g}_Lambda{land_l}'
                                    print(f'- select landscape: {random_land_idx} from folder {setting}')
                                    for j in tqdm(range(n), ncols=70):

                                        land_number = random_land_idx[j]
                                        file = os.path.join('landscape', setting, f'{setting}_{str(land_number)}.json')
                                        landscape = read_landscape(file, g)
                                        lmin = np.amin(landscape)
                                        lmax = np.amax(landscape)

                                        # select one iter output additional and decision map
                                        select_output_iter = -1
                                        if(is_output_decision_map or is_output_additional_output or is_output_first_decision_map):
                                            select_output_iter = random.randint(0, max_iter-1)
                                            print(f'\n- select one group output the decision map and the additional output: {select_output_iter}')
                                        
                                        # use a record list to store each map's result
                                        record_list = [Record(i, k, s) for i in range(group_cnt, group_cnt+max_iter)]
                                        group_cnt += max_iter
                                        thread_list = []

                                        for i in range(max_iter):
                                            # init record
                                            record_list[i].grid_size = g
                                            record_list[i].landscape_lambda = land_l
                                            record_list[i].agent_beta = b
                                            record_list[i].agent_alpha = a
                                            record_list[i].agent_lambda = l
                                            record_list[i].struct_type = struct_type
                                            record_list[i].structure = struct_name
                                            record_list[i].freqT = t
                                            record_list[i].min_max[0] = lmin
                                            record_list[i].min_max[1] = lmax
                                            record_list[i].land_number = f'{setting}_{str(land_number)}'
                                            record_list[i].iter_number = i
                                            record_list[i].fr_value = fr_value
                                            record_list[i].is_output_decision_map = is_output_decision_map
                                            record_list[i].is_output_additional_output = is_output_additional_output
                                            record_list[i].is_output_first_decision_map = is_output_first_decision_map

                                            if(i == select_output_iter):
                                                record_list[i].select_output_iter = True
                                                if(is_output_decision_map):
                                                    map_folder = os.path.join('decision_map', str(record_list[i].land_number))
                                                    os.makedirs(map_folder, exist_ok=True)
                                                    os.mkdir(os.path.join(map_folder, str(record_list[i].group_number)))

                                                if(is_output_first_decision_map):
                                                    map_folder = os.path.join('decision_map', 'first_round')
                                                    os.makedirs(map_folder, exist_ok=True)
                                                    map_folder = os.path.join(map_folder, str(record_list[i].land_number))
                                                    os.makedirs(map_folder, exist_ok=True)
                                                    os.mkdir(os.path.join(map_folder, str(record_list[i].group_number)))

                                            thread_list.append(threading.Thread(target=one_game_job, args=(args, record_list[i], weights, landscape, setting)))
                                            thread_list[i].start()

                                        for i in range(max_iter):
                                            thread_list[i].join()
                                            if(record_list[i].select_output_iter and record_list[i].is_output_additional_output):
                                                record_list[i].export_additional_output()
                                            record_list[i].write()

    print("Total time: ", time()-starttime)
