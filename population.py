"""
population :
1. individual : initialize, update
2. population :initialize, update
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

# P T types
act_type_dict = {1: "linear", 2: "elu", 3: "sigmoid", 4: "tanh", 5: "relu", 6: "relu6", 7: "softplus", 8: "leaky_relu"}
# agg_type_dict = {1: "Graph", 2: "SplineFeaturePropagation", 3: "SAGEFeaturePropagation", 4: "GATFeaturePropagation"}
agg_type_dict = {1: "Graph", 2: "SplineFeaturePropagation", 3: "SAGEFeaturePropagation"}

class IndividualGNN(object):
    def __init__(self, args, adj, edge_index, edge_attr):
        self.args = args
        self.adj = adj
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def initialize_individual(self):
        individual = {}
        # Step 1: Randomly generate the GNN structure order_type represented by the numbers 1 and 2 for the P and T layers
        num_layers = np.random.randint(self.args.min_layers, self.args.max_layers + 1)
        order_type = np.random.choice([1, 2], size=num_layers).tolist()


        inchannels = [None for _ in range(num_layers)]
        outchannels = [None for _ in range(num_layers)]
        prop_type = [None for _ in range(num_layers)]
        agg_type = [None for _ in range(num_layers)]
        heads_num = [None for _ in range(num_layers)]
        act_type = [None for _ in range(num_layers)]

      
        encode_layers = []
        for i, layer_type in enumerate(order_type):
            if layer_type == 1: 
                inchannels[i] = self.args.hid_dim
                outchannels[i] = self.args.hid_dim
#                 selected_prop_type = np.random.choice([1, 2, 3, 4])
                selected_prop_type = np.random.choice([1,2,3])
                prop_type[i] = selected_prop_type
                if selected_prop_type == 1:
                    agg_type[i] = None
                    heads_num[i] = None
                    encode_layers.append(Graph(self.adj))
                elif selected_prop_type == 2:
                    agg_type[i] = None
                    heads_num[i] = None
                    encode_layers.append(SplineFeaturePropagation(self.args.hid_dim, self.args.hid_dim, self.args.dim, self.args.kernel_size, self.edge_index, self.edge_attr))
                elif selected_prop_type == 3:
                    selected_agg_type = np.random.choice([1, 2, 3])
                    agg_type[i] = selected_agg_type
                    heads_num[i] = None
                    encode_layers.append(SAGEFeaturePropagation(self.args.hid_dim, self.args.hid_dim, aggregator=selected_agg_type))

            elif layer_type == 2:  
                selected_act_type = np.random.choice(list(act_type_dict.keys()))
                act_type[i] = selected_act_type
                encode_layers.append(MLP(self.args.hid_dim, self.args.hid_dim, self.args.mdrop, act_type=selected_act_type))

        # Step 4: Store the generated properties in the individual dictionary and return this dictionary
        individual['inchannels'] = inchannels
        individual['outchannels'] = outchannels
        individual['prop_type'] = prop_type
        individual['agg_type'] = agg_type
        individual['act_type'] = act_type
        individual['heads_num'] = heads_num
        individual['acc'] = -1.0
        individual['f1'] = -1.0

        individual['model_size(MB)_acc'] =  -1.0
        individual['param_count(MB)_acc'] =  -1.0
        individual['model_size(MB)_f1'] =  -1.0
        individual['param_count(MB)_f1'] =  -1.0
        individual['encode_layers'] = encode_layers
        individual['order_type'] = order_type

        individual['best_para_acc'] = None
        individual['best_para_f1'] = None


        return individual

    def update_offspring_information(self, offsprings):
        for off in offsprings:
            order_type = off['order_type']
            prop_type = off['prop_type']
            agg_type = off['agg_type']
            act_type = off['act_type']
            encode_layers = []
            for i, layer_type in enumerate(order_type):
                if layer_type == 1:
                    if prop_type[i] == 1:
                        encode_layers.append(Graph(self.adj))
                    elif prop_type[i] == 2:
                        encode_layers.append(SplineFeaturePropagation(self.args.hid_dim, self.args.hid_dim, self.args.dim, self.args.kernel_size, self.edge_index, self.edge_attr))
                    elif prop_type[i] == 3:
                        encode_layers.append(SAGEFeaturePropagation(self.args.hid_dim, self.args.hid_dim, aggregator=agg_type[i]))
                elif layer_type == 2:
                    encode_layers.append(MLP(self.args.hid_dim, self.args.hid_dim, self.args.mdrop, act_type=act_type[i]))
            off['encode_layers'] = encode_layers
        return offsprings


class PopulationGNN(object):
    def __init__(self, args, adj, edge_index, edge_attr):
        self.args = args
        self.adj = adj
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.individuals = []

    def initialize_individuals(self):
        init_individual = IndividualGNN(self.args, self.adj, self.edge_index, self.edge_attr)
        for _ in range(self.args.pop_size):
            individual = init_individual.initialize_individual()
            self.individuals.append(individual)
        return self.individuals

    def create_from_offspring(self, offsprings):
        off_individual = IndividualGNN(self.args, self.adj, self.edge_index, self.edge_attr)
        offsprings = off_individual.update_offspring_information(offsprings)
        return offsprings
    
class PopulationGNN(object):
    def __init__(self, args, adj, edge_index, edge_attr):
        self.args = args
        self.adj = adj
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.individuals = []

    def initialize_individuals(self):
        init_individual = IndividualGNN(self.args, self.adj, self.edge_index, self.edge_attr)
        for _ in range(self.args.pop_size):
            individual = init_individual.initialize_individual()
            self.individuals.append(individual)
        return self.individuals

    def create_from_offspring(self, offsprings):
        off_individual = IndividualGNN(self.args, self.adj, self.edge_index, self.edge_attr)
        offsprings = off_individual.update_offspring_information(offsprings)
        return offsprings
