import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from samplePrior2D import plot_kernel
import os
import copy
import matplotlib.pyplot as plt
import json

def cal_credibility(actual_rewards):
    credit = {
        'sum': np.sum(actual_rewards), 
        'mean': np.mean(actual_rewards), 
        'max': np.max(actual_rewards), 
        'median': np.median(actual_rewards)
    }
    return credit

struct_types = {'windfall': 1, 'merit-based': 2}
structures = {'P': 1, 'IT': 2, 'T': 3, 'V': 4}
file_path = {'mapping': 'output/group_mapping.csv', 'group': 'output/group_decision.csv', 'agent': 'output/agent_decision.csv'}
original_value_file_path = {'group': 'output/group_decision(original_value).csv', 'agent': 'output/agent_decision(original_value).csv'}
class Record():
    def __init__(self, group, k, s, is_scale=False):
        # ********** init params variable **************
        self.group_number = group
        self.grid_size = 0
        self.landscape_lambda = 0
        # self.agent_beta = []
        self.agent_alpha = []
        self.agent_lambda = 0
        self.struct_type = 0
        self.structure = 0
        self.freqT = 0
        self.k = k
        self.s = s

        self.land_number = 0
        self.iter_number = 0
        self.select_output_iter = True

        # ****** fit scaler at the end of this group *******
        self.min_max = [float('inf'), float('-inf')]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.is_scale = is_scale

        # *********** df records for each round ************
        self.group_mapping_df = pd.DataFrame()
        self.group_decisions_df = pd.DataFrame()
        self.agent_decisions_df = pd.DataFrame()

        # additional output
        self.agent_imagine_per_round = {}   # {round: imagine_landscape, ...}
        self.agent_GP_per_round = {}
        self.agent_observations_per_round = {}

        # flag
        self.is_output_decision_map = True
        self.is_output_additional_output = True
        self.is_output_first_decision_map = True

    def group_decision(self, round, selects, credit):
        new_row = {}
        new_row['group_number'] = self.group_number
        new_row['round'] = round
        new_row['selected_number'] = len(selects)

        positions = list(selects.keys())
        rewards = np.round(list(selects.values()), 2)
        for i in range(self.k):
            if(i < len(selects)):
                new_row['position'+str(i)] = positions[i]
            else:
                new_row['position'+str(i)] = '-'
        for i in range(self.k):
            if(i < len(selects)):
                new_row['value'+str(i)] = rewards[i]
            else:
                new_row['value'+str(i)] = np.nan

        new_row['group_sum'] = np.round(credit['sum'], 2)
        new_row['group_max'] = np.round(credit['max'], 2)
        new_row['group_mean'] = np.round(credit['mean'], 2)
        new_row['group_median'] = np.round(credit['median'], 2)

        self.group_decisions_df = pd.concat([self.group_decisions_df, pd.DataFrame([new_row])], ignore_index=True)

    def agent_decision(self, round, agent, actual_rewards,c_strategy=None):
        new_row = {}
        new_row['group_number'] = self.group_number
        new_row['agent_number'] = agent.agent_number
        # new_row['agent_beta'] = agent.beta
        new_row['agent_alpha'] = agent.alpha
        new_row['agent_lambda'] = agent.l
        new_row['agent_position'] = agent.position
        new_row['selected_number'] = agent.weight
        new_row['round'] = round

        positions = agent.proposals
        actual_rewards = np.round(actual_rewards, 2)
        exploit_prob = agent.exploit_probability

        proposal_k = np.min([len(agent.proposals), self.k])

        for i in range(proposal_k):
            new_row['position'+str(i)] = positions[i]
        for i in range(proposal_k):
            new_row['value'+str(i)] = actual_rewards[i]
        # We don't have the probability for the last one
        if round<1:
            for i in range(proposal_k-1): 
                new_row[f'Exploitation probability {i}'] = exploit_prob[i]
        else:
            new_row[f'Exploitation probability for group decision'] = exploit_prob[-1]
        
        new_row['agent_sum'] = np.round(agent.credit['sum'], 2)
        new_row['agent_max'] = np.round(agent.credit['max'], 2)
        new_row['agent_mean'] = np.round(agent.credit['mean'], 2)
        new_row['agent_median'] = np.round(agent.credit['median'], 2)

        new_row['agent_observ_sum'] = np.round(agent.credit_observation['sum'], 2)
        new_row['agent_observ_max'] = np.round(agent.credit_observation['max'], 2)
        new_row['agent_observ_mean'] = np.round(agent.credit_observation['mean'], 2)
        new_row['agent_observ_median'] = np.round(agent.credit_observation['median'], 2)

        if(round % self.freqT == 0) and (round > 0):
            if len(agent.observation_values_for_turnover)==0:
                agent.observation_values_for_turnover = [-1]
            turnover_cred = cal_credibility(agent.observation_values_for_turnover)
            new_row[f'turnover_{c_strategy}'] = np.round(turnover_cred[c_strategy], 2)
        
        self.agent_decisions_df = pd.concat([self.agent_decisions_df, pd.DataFrame([new_row])], ignore_index=True)

    def update_group_mapping(self, land_number, n):
        new_row = {}
        new_row['group_number'] = self.group_number
        new_row['land_number'] = land_number
        new_row['iter_number'] = self.iter_number
        new_row['grid'] = self.grid_size * self.grid_size
        new_row['landscape_lambda'] = self.landscape_lambda
        # new_row['agent_betas'] = self.agent_beta
        new_row['agent_alpha'] = self.agent_alpha
        new_row['agent_lambda'] = self.agent_lambda
        new_row['struct_type'] = struct_types[self.struct_type]
        new_row['hierarchical_structure'] = structures[self.structure]
        new_row['freqT'] = self.freqT
        new_row['sigmoid'] = self.s
        new_row['fr_value'] = self.fr_value
        new_row['round'] = n
        new_row['select_output'] = self.select_output_iter
        self.group_mapping_df = pd.concat([self.group_mapping_df, pd.DataFrame([new_row])], ignore_index=True)


    def write(self):
        # *** write setting ***
        mode = 'w' if(self.group_number == 0) else 'a'
        header = True if(self.group_number == 0) else False
        # **************** min_max scale for values *******************
        if(self.is_scale):
            # group decision
            scaled_group_decisions_df = copy.deepcopy(self.group_decisions_df)
            for i in range(self.k):
                if(len(self.group_decisions_df) == 0):
                    break
                rows = np.asarray(list(scaled_group_decisions_df['value'+str(i)]))
                rows_2D = self.scaler.transform(np.reshape(rows, (-1,1)))
                rows = (np.array([i[0] for i in rows_2D])).tolist()
                rows = np.round(rows, 2)
                scaled_group_decisions_df['value'+str(i)] = rows
            
            # agent decision
            scaled_agent_decisions_df = copy.deepcopy(self.agent_decisions_df)
            for i in range(self.k):
                rows = np.asarray(list(scaled_agent_decisions_df['value'+str(i)]))
                rows_2D = self.scaler.transform(np.reshape(rows, (-1,1)))
                rows = (np.array([i[0] for i in rows_2D])).tolist()
                rows = np.round(rows, 2)
                scaled_agent_decisions_df['value'+str(i)] = rows
            
            # credit
            for _, row in scaled_group_decisions_df.iterrows():
                actual_rewards = [row['value'+str(i)] for i in range(self.k) if str(row['value'+str(i)]) != 'nan']
                actual_rewards = np.round(actual_rewards, 2)
                scaled_group_decisions_df.loc[_, 'group_sum'] = np.round(np.sum(actual_rewards), 2)
                scaled_group_decisions_df.loc[_, 'group_mean'] = np.round(np.mean(actual_rewards), 2)
                scaled_group_decisions_df.loc[_, 'group_max'] = np.round(np.max(actual_rewards), 2)
                scaled_group_decisions_df.loc[_, 'group_median'] = np.round(np.median(actual_rewards), 2)
            for _, row in scaled_agent_decisions_df.iterrows():
                actual_rewards = [row['value'+str(i)] for i in range(self.k)]
                actual_rewards = np.round(actual_rewards, 2)
                scaled_agent_decisions_df.loc[_, 'agent_sum'] = np.round(np.sum(actual_rewards), 2)
                scaled_agent_decisions_df.loc[_, 'agent_mean'] = np.round(np.mean(actual_rewards), 2)
                scaled_agent_decisions_df.loc[_, 'agent_max'] = np.round(np.max(actual_rewards), 2)
                scaled_agent_decisions_df.loc[_, 'agent_median'] = np.round(np.median(actual_rewards), 2)
            # *** write to file ***
            scaled_group_decisions_df.fillna(value='-', inplace=True)
            scaled_group_decisions_df.to_csv(file_path['group'], mode=mode, index=False, header=header)
            scaled_agent_decisions_df.to_csv(file_path['agent'], mode=mode, index=False, header=header)

        self.group_decisions_df.fillna(value='-', inplace=True)
        self.group_mapping_df.to_csv(file_path['mapping'], mode=mode, index=False, header=header)
        self.group_decisions_df.to_csv(original_value_file_path['group'], mode=mode, index=False, header=header)
        self.agent_decisions_df.to_csv(original_value_file_path['agent'], mode=mode, index=False, header=header)     
    
    def print_setting(self):
        print(f'- grid size: {self.grid_size}*{self.grid_size}')
        print(f'- landscape lambda: {self.landscape_lambda}')
        # print(f'- agent beta: {self.agent_beta}')
        print(f'- agent alpha: {self.agent_alpha}')
        print(f'- agent lambda: {self.agent_lambda}')
        print(f'- structure: {self.structure} ({self.struct_type})')
        print(f'- freqT: {self.freqT}')
        print(f'- (min, max): ({self.min_max[0]}, {self.min_max[1]})')
    
    def get_scaler(self):
        if(self.is_scale):
            self.scaler.fit(np.asarray(np.reshape([self.min_max[0], self.min_max[1]], (-1, 1))))
    
    def export_additional_output(self):
        folder = os.path.join('additional_output', 'map'+str(self.land_number))
        os.makedirs(folder, exist_ok=True)

        folder = os.path.join('additional_output', 'map'+str(self.land_number), 'group'+str(self.group_number))
        os.makedirs(folder, exist_ok=True)
        
        for _, imagine_map in self.agent_imagine_per_round.items():
            imageFileName = os.path.join(folder, f'{str(_)}.png')

            s = np.asarray(imagine_map)
            # if(self.is_scale):
            #     s_2D = self.scaler.transform(np.reshape(s, (-1,1)))
            #     s = np.array([i[0] for i in s_2D])

            observations = self.agent_observations_per_round[_]
            plot_kernel(imageFileName, s, self.grid_size, observations)

            jsonFileName = os.path.join(folder, f'{str(_)}.json')
            s_list = s.tolist()
            with open(jsonFileName, 'w') as json_file:
                json.dump(s_list,json_file)
            print(f'wrote json at {jsonFileName}')


class Beta_greedy_record():
    def __init__(self, betas):
        self.betas = betas
        self.ratio = [[] for j in range(10)]
        for i in range(10):
            self.ratio[i] = [0 for j in range(len(self.betas))]
        
        self.overlap_df = pd.DataFrame()

    def cal_overlap(self, r, b_idx, proposals):
        for pos in proposals[0]:
            if(pos in proposals[1]):
                self.ratio[r][b_idx] += 1
        self.ratio[r][b_idx] /= len(proposals[0])

        # print(self.ratio[r])
    
    def write_png(self):
        y = self.ratio
        x = [self.betas for _ in range(len(y))]
        for i in range(10):
            plt.plot(x[i], y[i], 'o-')    # 先畫第一條紅色實線
        plt.xticks(np.linspace(5,100,20))
        plt.yticks(np.linspace(0,1,11)) 
        plt.legend([r for r in range(1, 11)])
        plt.savefig('beta_test.png')

        # row: each round
        # col: each beta
        for i in range(10):
            self.overlap_df['round'+str(i)] = y[i]
        self.overlap_df.to_csv('overlap.csv', index=False)
    
    def overlap_per_beta(self):
        y = copy.deepcopy(self.ratio)
        all_round_per_beta = []
        for i in range(len(self.betas)):
            all_round_per_beta.append([y[j][i] for j in range(len(y))])
        
        return all_round_per_beta
        
        
