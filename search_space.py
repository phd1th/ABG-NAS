"""
search space :
combine P,T
P:
type =1 : Graph
type =2 : SplineFeaturePropagation
type =3 : GraphSAGE

T :
activations = ["linear", "elu", "sigmoid", "tanh", "relu", "relu6", "softplus", "leaky_relu"]
"""     
import sys
import time
import torch
import torch.utils
import torch.nn.functional as F
from torch import nn
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

# type =1 : Graph

class Graph(nn.Module):
    def __init__(self, adj,aggregator=-1,heads=-1):
        super(Graph, self).__init__()
        self.adj = adj
        self.aggregator = aggregator 
        self.heads = heads

    def forward(self, x):
        x = self.adj.matmul(x)
        return x
    
#type =2 : SplineFeaturePropagation
class SplineFeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size, edge_index, edge_attr,aggregator=-1,heads=-1):
        super(SplineFeaturePropagation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()
        self.aggregator = aggregator 
        self.heads = heads
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        B = self.compute_b_spline_kernel(edge_attr)
        out = torch.zeros_like(x)
        for i in range(row.size(0)):
            out[row[i]] += B[i] * x[col[i]]
        return out
    
    # #####corafull
    # def forward(self, x, edge_index, edge_attr):
    #     if edge_attr is None and self.edge_attr is None:
    #          edge_attr = torch.ones(edge_index.size(1)).to(x.device)
    #     elif edge_attr is None:
    #         edge_attr = self.edge_attr   
    #     row, col = edge_index
    #     B = self.compute_b_spline_kernel(edge_attr)
    #     out = torch.zeros_like(x)
    #     for i in range(row.size(0)):
    #         out[row[i]] += B[i] * x[col[i]]
    #     return out
    
    def compute_b_spline_kernel(self, edge_attr):
        return torch.exp(-edge_attr)
    
 

#  type =3 : GraphSAGE 
# aggregator =[mean, max, sum]
#op =SAGEFeaturePropagation(in_channels, out_channels, aggregator)
class SAGEFeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels, aggregator=1,heads=-1):
        super(SAGEFeaturePropagation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator  # 1 for mean, 2 for max, 3 for sum
        self.linear = nn.Linear(in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.reset_parameters()
        self.heads = heads
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        out = torch.zeros_like(x)
        
        if self.aggregator == 1:  # mean
            out.index_add_(0, row, x[col])
            deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1)
            out = out / deg.view(-1, 1)
        elif self.aggregator == 2:  # max
            for i in range(x.size(0)):
                neighbors = col[row == i]
                if neighbors.numel() > 0:
                    out[i] = torch.max(x[neighbors], dim=0)[0]
        elif self.aggregator == 3:  # sum
            out.index_add_(0, row, x[col])
        
        out = self.linear(out)
        out = torch.matmul(out, self.weight)
        return out

# type =4 : GAT , one head/ multi
#op = GATConvLayer(in_channels, hidden_channels, heads, concat, dropout,aggregator)

class GATFeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,aggregator=-1):
        super(GATFeaturePropagation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator 
        self.heads = heads
        self.concat = concat
        
        # Define the parameters for multiple heads
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
    
    def forward(self, x, edge_index):
        # Apply the linear transformation and reshape for multiple heads
        x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)
        
        row, col = edge_index
        alpha_l = (x[row] * self.attn_l).sum(dim=-1)
        alpha_r = (x[col] * self.attn_r).sum(dim=-1)
        alpha = F.leaky_relu(alpha_l + alpha_r)
        alpha = F.softmax(alpha, dim=1)
        
        # Initialize output tensor
        out = torch.zeros_like(x)
        
        # Aggregate features for each node
        for i in range(x.size(0)):
            out[i] = torch.sum(alpha[i].unsqueeze(-1) * x[col[i]], dim=0)
        
        if self.concat:
            return out.view(-1, self.heads * self.out_channels)
        else:
            return out.mean(dim=1)
        
"""

activations = ["linear", "elu", "sigmoid", "tanh", "relu", "relu6", "softplus", "leaky_relu"]
act_type=[1,2,3,4,5,6,7,8]
1: linear
2: ELU
3: Sigmoid
4: Tanh
5: ReLU
6: ReLU6
7: Softplus
8: Leaky ReLU
"""

class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout, act_type="relu", last=False):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout
        self.act_type = act_type
        self.last = last

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        if not self.last:
            x = self.activate(x, self.act_type)
        return x
# act_type=[1,2,3,4,5,6,7,8]
    def activate(self, x, act_type):
        if act_type == 1:
            # print("Activation: linear")
            return x
        elif act_type == 2:
            # print("Activation: elu")
            return F.elu(x)
        elif act_type == 3:
            # print("Activation: sigmoid")
            return torch.sigmoid(x)
        elif act_type == 4:
            # print("Activation: tanh")
            return torch.tanh(x)
        elif act_type == 5:
            # print("Activation: relu")
            return F.relu(x)
        elif act_type == 6:
            # print("Activation: relu6")
            return F.relu6(x)
        elif act_type == 7:
            # print("Activation: softplus")
            return F.softplus(x)
        elif act_type == 8:
            # print("Activation: leaky_relu")
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unknown activation type: {act_type}")
