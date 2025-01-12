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
import os
from utils import *
from torch_geometric.datasets import CoraFull
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import train_test_split
"""
    Load data for the given dataset for train, validation, and test sets.
"""
# cora/pubmed
def load_data(path="your_datapath/data", dataset="cora"):
 
    print("\n[STEP 1]: Loading {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for name in names:
        with open(os.path.join(path, f"ind.{dataset}.{name}"), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(path, f"ind.{dataset}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    if dataset == 'citeseer':
        save_label = np.where(labels)[1]

    labels = torch.LongTensor(np.where(labels)[1])

    # Define indexes for train, validation, and test
    num_all = len(labels)
    num_train = int(num_all * 0.6)
    num_val = int(num_all * 0.2)

    idx_train = range(num_train)
    idx_val = range(num_train, num_train + num_val)
    idx_test = range(num_train + num_val, num_all)

    idx_train = torch.LongTensor(list(idx_train))
    idx_val = torch.LongTensor(list(idx_val))
    idx_test = torch.LongTensor(list(idx_test))

    if dataset == 'citeseer':
        labels = np.insert(save_label, test_idx_reorder, labels.numpy(), axis=0)
        labels = torch.LongTensor(labels)

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    # 提取 edge_index 和 edge_attr
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    edge_attr = torch.tensor(adj.data, dtype=torch.float)

    return adj, features, labels, idx_train, idx_val, idx_test, edge_index, edge_attr

#citeseer
def edge_index_to_adj(edge_index, num_nodes):
    row, col = edge_index
    adj = sp.coo_matrix((torch.ones(row.size(0)).cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                        shape=(num_nodes, num_nodes), dtype=float)
    return adj


def load_data(dataset="citeseer"):
    dataset = torch.load('your_datapath')
    data = dataset[0]
    features = data.x
    labels = data.y

    num_nodes = data.num_nodes

    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    test_size = num_nodes - train_size - val_size

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True
    
    edge_index = data.edge_index
    
    adj = edge_index_to_adj(edge_index, data.num_nodes)
    # G = nx.from_scipy_sparse_matrix(adj)
    # citerseer part
    G = nx.from_scipy_sparse_array(adj)

    adj = nx.adjacency_matrix(G)
    edge_attr = torch.tensor(adj.data, dtype=torch.float) if adj.data.size else None
    
    return adj, features, labels, train_mask, val_mask, test_mask, edge_index, edge_attr


#corafull
def load_data():
    # dataset = CoraFull(root='/tmp/CoraFull', transform=NormalizeFeatures())
    dataset = CoraFull(root='your_datapath', transform=NormalizeFeatures())
    data = dataset[0]
    
    adj = to_scipy_sparse_matrix(data.edge_index)
    adj = aug_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = data.x
    labels = data.y

    indices = np.arange(len(labels))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=42)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    edge_index = data.edge_index
    edge_attr = None  # Not used in this example

    return adj, features, labels, idx_train, idx_val, idx_test, edge_index, edge_attr
