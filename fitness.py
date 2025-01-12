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

"""
1. ModelOp
2. decode 
3. train& valid & test
"""     
class ModelOp(nn.Module):
    
    def __init__(self, individual, adj, feat_dim, hid_dim, num_classes, fdropout, mdropout, dropout, dim, kernel_size, edge_index, edge_attr, device):
        super(ModelOp, self).__init__()
        self.individual = individual 
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = individual['order_type']
        self.edge_index = edge_index.to(device)
        self.edge_attr = edge_attr.to(device)
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.adj = adj.to(device)
        self.dim = dim
        self.kernel_size = kernel_size
        self.device = device
        
        for i, layer_type in enumerate(self._arch):
            if layer_type == 1:
                prop_type = individual['prop_type'][i]
                # print('prop_type:',prop_type)
                if prop_type == 1:
                    op = Graph(self.adj).to(device)
                elif prop_type == 2:
                    op = SplineFeaturePropagation(hid_dim, hid_dim, dim, kernel_size, self.edge_index, self.edge_attr).to(device)
                elif prop_type == 3:
                    op = SAGEFeaturePropagation(hid_dim, hid_dim, aggregator=individual['agg_type'][i]).to(device)
                else:
                    print("prop_type wrong！！！！")
                # elif prop_type == 4:
                #     op = GATFeaturePropagation(hid_dim, hid_dim, heads=individual['heads_num'][i]).to(device)
                self._numP += 1
            elif layer_type == 2:
                act_type = individual['act_type'][i]
                op = MLP(hid_dim, hid_dim, mdropout, act_type).to(device)
            else:
                raise ValueError("Invalid layer type")
            self._ops.append(op)

        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True).to(device)
        self.linear = MLP(feat_dim, hid_dim, fdropout, act_type=1, last=True).to(device)
        self.classifier = MLP(hid_dim, num_classes, dropout, act_type=1, last=True).to(device)

    def forward(self, data):
        s0 = data['x'].to(self.device)
        res = self.linear(s0)
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                tempP.append(res)
                numP.append(i)
                totalP += 1
                if self._arch[i] == 1:
                    if self.individual['prop_type'][i] == 2:  # SplineFeaturePropagation requires additional inputs
                        res = self._ops[i](res, self.edge_index, self.edge_attr)
                    elif self.individual['prop_type'][i] ==3:  # SAGEFeaturePropagation and GATFeaturePropagation require edge_index
                        res = self._ops[i](res, self.edge_index)
                    # elif self.individual['prop_type'][i] ==4:  # SAGEFeaturePropagation and GATFeaturePropagation require edge_index
                    #     res = self._ops[i](res, self.edge_index)
                    else:
                        res = self._ops[i](res)
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    res = self._ops[i](res)
                    tempT.append(res)
                    numP = []
                    tempP = []
            else:
                if self._arch[i - 1] == 1:
                    if self._arch[i] == 2:
                        res = sum([torch.mul(torch.sigmoid(self.gate[totalP - len(numP) + j]), tempP[j]) for j in range(len(numP))])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    else:
                        if self.individual['prop_type'][i] == 2:
                            res = self._ops[i](res, self.edge_index, self.edge_attr)
                        elif self.individual['prop_type'][i] ==3:
                            res = self._ops[i](res, self.edge_index)
                        else:
                            res = self._ops[i](res)
                        tempP.append(res)
                        numP.append(i - point)
                        totalP += 1
                else:
                    if self._arch[i] == 1:
                        if self.individual['prop_type'][i] == 2:
                            res = self._ops[i](res, self.edge_index, self.edge_attr)
                        elif self.individual['prop_type'][i] ==3:
                            res = self._ops[i](res, self.edge_index)
                        else:
                            res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)
        if len(numP) > 0 or len(tempP) > 0:
            res = sum([torch.mul(torch.sigmoid(self.gate[totalP - len(numP) + j]), tempP[j]) for j in range(len(numP))])
        else:
            res = sum(tempT)
        logits = self.classifier(res)
        logits = F.log_softmax(logits, dim=1)
        return logits
    

"""fitness evaluate"""
# Train function
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data['train_mask']], data['y'][data['train_mask']])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate function
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        loss = F.nll_loss(out[mask], data['y'][mask]).item()
        acc = pred[mask].eq(data['y'][mask]).sum().item() / int(mask.numel())
        # # citerseer
        # acc = pred[mask].eq(data['y'][mask]).sum().item() / int(mask.sum())
        f1 = f1_score(data['y'][mask].cpu(), pred[mask].cpu(), average='macro')
        cm = confusion_matrix(data['y'][mask].cpu(), pred[mask].cpu())
    return loss, acc, f1, cm

#decode
def decode_architecture(order_type, encoder_layers):
    # Define the mapping for P types
    P_types = {
        1: "Graph",
        2: "SplineFeaturePropagation",
        3: "GraphSAGE",
        # 4: "GAT"
    }
    
    # Define the mapping for T types (activations)
    T_types = {
        1: "linear",
        2: "ELU",
        3: "Sigmoid",
        4: "Tanh",
        5: "ReLU",
        6: "ReLU6",
        7: "Softplus",
        8: "Leaky ReLU"
    }
    
    # Start building the decoded architecture string
    decoded_arch = []
    
    # Iterate through order_type and encoder_layers to decode the architecture
    for i, layer_type in enumerate(order_type):
        if layer_type == 1:  # This is a P (propagation) layer
            prop_type = encoder_layers['prop_type'][i]
            if prop_type in P_types:
                if prop_type == 3:  # GraphSAGE needs aggregator info
                    agg_type = encoder_layers.get('agg_type', [None] * len(order_type))[i]
                    aggregator = {1: "mean", 2: "max", 3: "sum"}.get(agg_type, "unknown")
                    decoded_arch.append(f"P({P_types[prop_type]}:{aggregator})")
                elif prop_type == 4:  # GAT needs heads_num info
                    heads_num = encoder_layers.get('heads_num', [None] * len(order_type))[i]
                    decoded_arch.append(f"P({P_types[prop_type]}:{heads_num} heads)")
                else:  # Other types of P
                    decoded_arch.append(f"P({P_types[prop_type]})")
        elif layer_type == 2:  # This is a T (transformation) layer
            act_type = encoder_layers['act_type'][i]
            if act_type in T_types:
                decoded_arch.append(f"T({T_types[act_type]})")
    
    # Join the decoded architecture parts with '-' and return
    return "arch= [" + '-'.join(decoded_arch) + "]"

#corafull
def train_and_eval(args, population, data, index, gen_now, mutation_flag, best_gen):
    best_val_f1 = 0
    best_model_path_f1 = './best_model_f1.pth'
    best_individual_f1 = None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    adj, features, labels, edge_index, edge_attr = data

    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    #cora/pubmed
    edge_attr = edge_attr.to(device)
    # #corafull &citeseer
    # edge_attr = edge_attr.to(device) if edge_attr is not None else None

    for count, individual in enumerate(population, 1):
        if mutation_flag == 0:
            print(f"The {count}st individual from {gen_now}st generation population will be evaluated")
        elif mutation_flag == 1:
            print(f"The {count}st individual from {gen_now}st population after mutation will be evaluated")
        elif float(individual['f1']) > 0:
            print(f"The {count}st individual from {gen_now}st generation population without involving in mutation process don't need evaluate ")
            continue

        arch = individual['order_type']
        print(f'Training individual {count} with architecture: {arch}')

        model = ModelOp(individual, adj, features.shape[1], args.hid_dim, len(torch.unique(labels)), args.fdrop, args.mdrop, args.drop, args.dim, args.kernel_size, edge_index, edge_attr, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_individual_val_f1 = 0
        trigger_times = 0
        patience = 30
        train_losses = []
        val_losses = []

        for epoch in range(args.epochs):
            train_loss = train(model, optimizer, {'x': features, 'y': labels, 'train_mask': index[0].to(device), 'val_mask': index[1].to(device), 'test_mask': index[2].to(device)})
            val_loss, val_acc, val_f1, _ = evaluate(model, {'x': features, 'y': labels}, index[1].to(device))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Accuracy {val_acc:.4f}, F1 Score {val_f1:.4f}')

            if val_f1 > best_individual_val_f1:
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f'Early stopping for individual {count} at epoch {epoch+1}')
                    break
            
            if val_f1 > best_individual_val_f1:
                best_individual_val_f1 = val_f1
                individual['f1'] = best_individual_val_f1
                individual['best_para_f1'] = model.state_dict()

        if individual['f1'] > best_val_f1:
            best_val_f1 = individual['f1']
            best_individual_f1 = individual
            best_gen = gen_now

        print(f'Run {count} BEST Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Accuracy {val_acc:.4f}, F1 Score {best_individual_val_f1:.4f}')

        if not os.path.exists('./exp0_cora/plot'):
            os.makedirs('./exp0_cora/plot')

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.axvline(x=epoch - trigger_times, color='r', linestyle='--', label='Early Stopping Checkpoint')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for Run {count}')
        plt.legend()
        plt.savefig(f'./exp0_cora/plot/run_{count}_f1_{best_individual_val_f1}_results.png')
        plt.close()

    if best_individual_f1:
        print("Best individual by validation F1 score:")
        print(decode_architecture(best_individual_f1['order_type'], best_individual_f1))

    return population, best_model_path_f1, best_individual_f1, best_gen



def test_end(model_f1, data, best_individual_f1, curr_gen, all_test_results):
    print("Testing model with best validation F1 score:")
    model_f1.load_state_dict(best_individual_f1['best_para_f1'])
    model_f1.eval()
    with torch.no_grad():
        out = model_f1(data)
        pred = out.argmax(dim=1)
        test_acc = pred[data['test_mask']].eq(data['y'][data['test_mask']]).sum().item() / int(data['test_mask'].numel())
        test_f1 = f1_score(data['y'][data['test_mask']].cpu(), pred[data['test_mask']].cpu(), average='macro')
        test_cm = confusion_matrix(data['y'][data['test_mask']].cpu(), pred[data['test_mask']].cpu())

    print(f'Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}')
    print("Best model by validation F1 score structure:")
    print(decode_architecture(best_individual_f1['order_type'], best_individual_f1))

    plt.figure(figsize=(10, 7))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Confusion Matrix for Best Validation F1 Score Model')
    plt.savefig('./test_confusion_matrix_f1.png')
    plt.close()

    # save current results to[all_test_results] 
    arch = decode_architecture(best_individual_f1['order_type'], best_individual_f1)
    all_test_results.append((curr_gen, test_acc, test_f1, arch))
