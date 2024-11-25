"""
crossover and mutation 
1. crossover
2. mutation:  Four types of mutation operations: Add, Remove, Exchange, and Alter
3. selection : 1) tournamnet 2) tournament
"""
from search_space import *
import sys
import time
import torch
import torch.utils
import torch.nn.functional as F
from torch import nn
import random
import copy
import argparse
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import optuna
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

"""
1. crossover:
a.tournament selection parents N/2 times
b.do crossover exchange units after cross points
c. update encoder_layers's information after exchange
"""
"""
1.do N_current times crossover--according to prob_mu, do mutation in after crossover population
2. exploitation stage: do many times cro&mu
"""

#cro&mu
class CrossoverAndMutation(object):
    def __init__(self, individuals, args, N_current, k, exploration_phase, prob_crossover, prob_mutation):
        self.args = args
        self.individuals = individuals
        self.N_current = N_current
        self.k = k
        self.exploration_phase = exploration_phase
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.offsprings = []

    def process(self):
        crossover = CrossoverGNN(self.individuals, self.args, self.k, self.exploration_phase, self.prob_crossover)
        new_offspring_list = crossover.do_crossover()
        self.offsprings = new_offspring_list

        mutation = MutationGNN(self.offsprings, self.args, self.exploration_phase, self.N_current, self.prob_mutation)
        offsprings = mutation.do_mutation()

        return offsprings
    
class CrossoverGNN:
    def __init__(self, individuals, args, k, exploration_phase, prob_crossover):
        self.individuals = individuals
        self.args = args
        self.k = k
        self.exploration_phase = exploration_phase  # AMPGA
        self.prob_crossover = prob_crossover  # Use dynamically adjusted prob_crossover
        self.max_length = args.max_arch_length

    def _choose_one_parent(self):
        """Tournament selection with dynamic k"""
        count_ = len(self.individuals)
        indices = random.sample(range(count_), self.k)
        selected_individual = max(indices, key=lambda idx: self.individuals[idx]['f1'])
        return selected_individual



    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        

        while idx2 == idx1:
             
            idx2 = self._choose_one_parent()
        return idx1, idx2

    def cross_one(self, source1, source2, position1, position2):
        derive1_source = list(source1[:position1])
        derive2_source = list(source2[:position2])
        derive1_source.extend(source2[position2:])
        derive2_source.extend(source1[position1:])
        return derive1_source, derive2_source

    def reload_off_information(self, parent1_dict, parent2_dict, cp_x1, cp_x2, off1_sequence, off2_sequence):
        offspring1_dict, offspring2_dict = {}, {}
        off1_prop_type, off2_prop_type = self.cross_one(parent1_dict['prop_type'], parent2_dict['prop_type'], cp_x1, cp_x2)
        off1_agg_type, off2_agg_type = self.cross_one(parent1_dict['agg_type'], parent2_dict['agg_type'], cp_x1, cp_x2)
        off1_act_type, off2_act_type = self.cross_one(parent1_dict['act_type'], parent2_dict['act_type'], cp_x1, cp_x2)
        off1_heads_num, off2_heads_num = self.cross_one(parent1_dict['heads_num'], parent2_dict['heads_num'], cp_x1, cp_x2)

        offspring1_dict['prop_type'] = off1_prop_type
        offspring1_dict['agg_type'] = off1_agg_type
        offspring1_dict['act_type'] = off1_act_type
        offspring1_dict['heads_num'] = off1_heads_num
        offspring1_dict['acc'] = -1.0
        offspring1_dict['f1'] = -1.0
        offspring1_dict['model_size(B)'] = -1.0
        offspring1_dict['encode_layers'] = []
        offspring1_dict['order_type'] = off1_sequence
        
        offspring2_dict['prop_type'] = off2_prop_type
        offspring2_dict['agg_type'] = off2_agg_type
        offspring2_dict['act_type'] = off2_act_type
        offspring2_dict['heads_num'] = off2_heads_num
        offspring2_dict['acc'] = -1.0
        offspring2_dict['f1'] = -1.0
        offspring2_dict['model_size(B)'] = -1.0
        offspring2_dict['encode_layers'] = []
        offspring2_dict['order_type'] = off2_sequence

        return offspring1_dict, offspring2_dict

    def do_crossover(self):
        new_offspring_list = []
        num_crossovers = len(self.individuals) // 2
        for _ in range(num_crossovers):
            ind1, ind2 = self._choose_two_diff_parents()
            parent1_dict, parent2_dict = copy.deepcopy(self.individuals[ind1]), copy.deepcopy(self.individuals[ind2])
            parent1, parent2 = parent1_dict['order_type'], parent2_dict['order_type']
            p_ = random.random()

            if p_ <= self.prob_crossover: 
                count_parent1, count_parent2 = len(parent1), len(parent2)
                alert_num = 0
                while True:
                    alert_num += 1
                    if alert_num > 10000:
                        break
                    cp_x1 = int(np.floor(np.random.random() * count_parent1))
                    cp_x2 = int(np.floor(np.random.random() * count_parent2))
                    if cp_x1 != 0 and cp_x2 != 0:
                        off1_sequence, off2_sequence = self.cross_one(parent1, parent2, cp_x1, cp_x2)
                        if len(off1_sequence) <= self.args.max_length and len(off2_sequence) <= self.args.max_length:
                            break
                if alert_num > 10000:
                    continue
                offspring1_dict, offspring2_dict = self.reload_off_information(parent1_dict, parent2_dict, cp_x1, cp_x2, off1_sequence, off2_sequence)
                new_offspring_list.append(offspring1_dict)
                new_offspring_list.append(offspring2_dict)
            else:
                count_parent1, count_parent2 = len(parent1), len(parent2)
                offspring1_dict, offspring2_dict = self.reload_off_information(parent1_dict, parent2_dict, count_parent1, count_parent2, parent1, parent2)
                new_offspring_list.append(offspring1_dict)
                new_offspring_list.append(offspring2_dict)
        return new_offspring_list
    

"""
2. mutation
    a. adding
    b.remove
    c.alter
    d.exchage
"""
act_type_dict = {1: "linear", 2: "elu", 3: "sigmoid", 4: "tanh", 5: "relu", 6: "relu6", 7: "softplus", 8: "leaky_relu"}
# agg_type_dict = {1: "Graph", 2: "SplineFeaturePropagation", 3: "SAGEFeaturePropagation", 4: "GATFeaturePropagation"}
agg_type_dict = {1: "Graph", 2: "SplineFeaturePropagation", 3: "SAGEFeaturePropagation"}


class MutationGNN:
    def __init__(self, individuals, args, exploration_phase, N_current, prob_mutation):
        self.individuals = individuals
        self.args = args
        self.exploration_phase = exploration_phase  # AMPGA
        self.N_current = N_current  # Add N_current parameter
        self.prob_mutation = prob_mutation  # Use dynamically adjusted prob_crossover


    def do_mutation(self):
        new_and_keep_off_list = []
        count_ = 0

        for individual in self.individuals:
            count_ += 1
            if self.exploration_phase:
                print('now is exploration stage!')
                
                # exploration stage: one time 
                p_ = random.random()
                if p_ <= self.prob_mutation:
                    new_and_keep_off_list.append(self.perform_mutation(individual))
                else:
                    new_and_keep_off_list.append(individual)
            else:
                # exploitation stage: 使用 N_current 次变异
                print('now is exploitation stage!')
                for _ in range(self.N_current):
                    individual = self.perform_mutation(individual)
                new_and_keep_off_list.append(individual)
        return new_and_keep_off_list

    def perform_mutation(self, individual):
        flag = 0
        while True:
            print('xxx--start select mu type--xxx')
            mutation_type = self.select_roulette(self.args.mutation_prob_list)
            if mutation_type == 0 and len(individual['order_type']) < self.args.max_length:
                flag = 1
            elif mutation_type == 1 and len(individual['order_type']) > 1:
                flag = 1
            elif mutation_type == 2:
                flag = 1
            elif mutation_type == 3 and len(individual['order_type']) > 1:
                flag = 1
            if flag == 1:
                break

        if mutation_type == 0:
            print('Do the ADD mutation')
            return self.add_unit(individual)
        elif mutation_type == 1:
            print('Do the REMOVE mutation')
            return self.remove_unit(individual)
        elif mutation_type == 2:
            print('Do the INNER ALTER mutation')
            return self.alter_unit(individual)
        elif mutation_type == 3:
            print('Do the EXCHANGE mutation')
            return self.exchange_unit(individual)

    
    def add_unit(self, individual):
        print('xxxxxxxstart addxxxxxxx')
        num_PU = 0
        for _type_ in individual['order_type']:
            if _type_ == 1:
                num_PU += 1

        # Randomly select between adding P or T
        select_type = np.random.choice([1, 2])

        while True:
            add_position = int(np.floor(np.random.random() * (len(individual['order_type']) - 2))) + 1
            
            if add_position == 0 & len(individual['order_type'])==1:
                add_position = 1  #20240701 修改：避免当len(individual['order_type'])=1时陷入死循环

            if select_type != 1 or add_position != 0:
                print("Add position is " + str(add_position))
                break

        if select_type == 1:
            print('Add P unit')
            prop_type = np.random.choice([1, 2, 3])
            agg_type = -1
            heads_num = -1

            if prop_type == 3:
                agg_type = np.random.choice([1, 2, 3])


            individual['order_type'].insert(add_position, 1)
            individual['prop_type'].insert(add_position, prop_type)
            individual['agg_type'].insert(add_position, agg_type)
            individual['heads_num'].insert(add_position, heads_num)
            individual['act_type'].insert(add_position, None)
            print('xxxxxxxend addxxxxxxx')
        
        elif select_type == 2:
            print('Add T unit')
            act_type = np.random.choice(list(act_type_dict.keys()))

            # Perform insert operation
            individual['order_type'].insert(add_position, 2)
            individual['prop_type'].insert(add_position, None)
            individual['agg_type'].insert(add_position, None)
            individual['heads_num'].insert(add_position, None)
            individual['act_type'].insert(add_position, act_type)
            print('xxxxxxxend addxxxxxxx')
        
        return individual

    

    def remove_unit(self, individual):
        print('xxxxxxxstart remxxxxxxx')
        while True:
         
            remove_position = int(np.floor(np.random.random() * len(individual['order_type'])))
            print("remove_position：",remove_position)
           
            if individual['order_type'][remove_position] != 1 or remove_position != 0:
                break

        print("Remove position is " + str(remove_position))
    
      

        del individual['order_type'][remove_position]
        del individual['prop_type'][remove_position]
        del individual['agg_type'][remove_position]
        del individual['heads_num'][remove_position]
        del individual['act_type'][remove_position]
        print('xxxxxxxend remxxxxxxx')
        return individual

    def exchange_unit(self, individual):
        print('xxxxxxxstart exchangexxxxxxx')
        length1 = len(individual['order_type'])
        length2 = len(individual['prop_type'])
        length3 = len(individual['agg_type'])
        length4 = len(individual['heads_num'])
        length5 = len(individual['act_type'])
        length6 = len(individual['encode_layers'])
        print ('length1:',length1)
        print ('length2:',length2)
        print ('length3:',length3)
        print ('length4:',length4)
        print ('length5:',length5)
        print ('length6:',length6)


        while True:
            
            pos_selected = np.random.randint(0, length1)
            pos_select = np.random.randint(0, length1)
            
            if length1==1:
                quit()
            if pos_selected != pos_select:
                break

        print(f"The selected mutant selected gene sequence position {pos_select} and position {pos_selected} for exchange")
        keys_to_exchange = [key for key in individual.keys() if isinstance(individual[key], list)]
        key_to_remove = 'encode_layers'
        if key_to_remove in keys_to_exchange:
            keys_to_exchange.remove(key_to_remove)
        print("test keys_to_exchange:",keys_to_exchange)
        for key in keys_to_exchange:
            individual[key][pos_select], individual[key][pos_selected] = individual[key][pos_selected], individual[key][pos_select]
        print('xxxxxxxend exchangexxxxxxx')
        return individual

    def alter_unit(self, individual):
        print('xxxxxxxstart alterxxxxxxx')
        alter_position = int(np.floor(np.random.random() * len(individual['order_type'])))
        print("Alter position is " + str(alter_position))

        if individual['order_type'][alter_position] == 1:  # P 层变异
            # alter_choice = np.random.choice(['type', 'aggregator', 'heads'])
            alter_choice = np.random.choice(['type', 'aggregator'])
            if alter_choice == 'type':
                original_type = individual['prop_type'][alter_position]
                # new_type = np.random.choice([1, 2, 3, 4])
                new_type = np.random.choice([1, 2, 3])

                if new_type == original_type:
                    # new_type = np.random.choice([1, 2, 3, 4])
                    new_type = np.random.choice([1, 2, 3])

                individual['prop_type'][alter_position] = new_type
                print(f"Changed P type from {original_type} to {new_type}")
            elif alter_choice == 'aggregator' and individual['prop_type'][alter_position] == 3:
                original_agg = individual['agg_type'][alter_position]
                new_agg = np.random.choice([1, 2, 3])
                if new_agg == original_agg:
                    new_agg = np.random.choice([1, 2, 3])
                individual['agg_type'][alter_position] = new_agg
                print(f"Changed aggregator type from {original_agg} to {new_agg}")

        elif individual['order_type'][alter_position] == 2:  # T 层变异
            original_act = individual['act_type'][alter_position]
            new_act = np.random.choice(list(act_type_dict.keys()))
            if new_act == original_act:
                new_act = np.random.choice(list(act_type_dict.keys()))
            individual['act_type'][alter_position] = new_act
            print(f"Changed activation function from {act_type_dict[original_act]} to {act_type_dict[new_act]}")
        print('xxxxxxxend alertxxxxxxx')
        return individual

    def select_roulette(self, _a):
        a = np.asarray(_a)
        k = 1
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float32)
        selected_index = []
        for i in range(k):
            u = np.random.rand() * sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ += sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index[0]
    


"""
3.selection:
a. environmental selection: roulette
b.  selection:  tournament(select 2)
"""
class Selection(object):
#1. Environmental_Selection : Roullette
    def RouletteSelection(self, _a, k):
        # _a = _a.cpu()
        # print('a:',_a)
        print('#')
        a = np.asarray(_a)
        # a = np.asarray([tensor.cpu() if tensor.is_cuda else tensor for tensor in _a])

        print('##')
        idx = np.argsort(a) 
        idx = idx[::-1] 
        sort_a = a[idx]
        print('###')
        sum_a = np.sum(a).astype(np.float32)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ += sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index


# 20240703 ONLY F1
def environment_selection(pops_parent, offs_newborn, args):
    v_f1_list = []
    indi_list = []

    for indi in pops_parent:
        indi_list.append(indi)
        v_f1_list.append(indi['f1'])

    for indi in offs_newborn:
        indi_list.append(indi)
        v_f1_list.append(indi['f1'])

    # print('v_f1_list:',v_f1_list)
    # v_f1_list = [tensor.cpu() if tensor.is_cuda else tensor for tensor in v_f1_list]
    # v_f1_list = [tensor.cpu() if isinstance(tensor, torch.Tensor) and tensor.is_cuda else tensor for tensor in v_f1_list]
    
    selection = Selection()
    combined_index_list = selection.RouletteSelection(v_f1_list, k=args.pop_size)

    max_f1_index = np.argmax(v_f1_list)

    if max_f1_index not in combined_index_list:
        f1_selected_v_list = [v_f1_list[i] for i in combined_index_list]
        min_f1_idx = np.argmin(f1_selected_v_list)
        combined_index_list[min_f1_idx] = max_f1_index

    print("len(combined_index_list):", len(combined_index_list))
    while len(combined_index_list) > args.pop_size:
        combined_index_list.pop()

    new_pops = [indi_list[i] for i in combined_index_list]

    return new_pops