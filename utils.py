

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


def calculate_crowding_distances(population):
    distances = np.zeros(len(population))
    for i in range(len(population[0]['order_type'])):
        sorted_population = sorted(population, key=lambda x: x['f1'])
        distances[0] = distances[-1] = float('inf')
        for j in range(1, len(population) - 1):
            distances[j] += (sorted_population[j + 1]['f1'] - sorted_population[j - 1]['f1'])
    return distances

def crowding_distance_selection(population, k):
    distances = calculate_crowding_distances(population)
    selected_indices = np.argsort(distances)[-k:]
    return [population[i] for i in selected_indices]


    
def calculate_avg_fitness_improvement(population):
    fitness_values = [individual['f1'] for individual in population]
    if len(fitness_values) <= 1:
        return 0
    fitness_diff = np.diff(fitness_values)
    avg_fitness_improvement = np.mean(fitness_diff)
    return avg_fitness_improvement

"""data and metrics"""
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.torch.sparse_coo_tensor(indices, values, shape)

def aug_normalized_adjacency(adj):
    """Normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def aug_random_walk(adj):
    """Random Walk algorithm."""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def count_parameters_in_MB(model):
# xxx(M) parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024.0 ** 2) * 4

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size / (1024.0 ** 2)  # Convert to MB

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# citerseer part

