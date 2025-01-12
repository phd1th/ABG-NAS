from search_space import *
from GA_operations import *
from utils import *
from population import *
from fitness import *
from dataset_load import *
from BO import *
import torch
import torch.utils
import torch.nn.functional as F
from torch import nn
import argparse
import numpy as np

"""
1.args setting
2. get data
3.evolution

"""


def setup_args():

# build ArgumentParser object
    parser = argparse.ArgumentParser(description="GNN Model Configuration")

    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--hid_dim', type=int, default=128, help='hidden dims')
    parser.add_argument('--hiddim', type=int, default=128, help='hidden dims')
    parser.add_argument('--fdrop', type=float, default=0.4, help='drop for cora feature')
    parser.add_argument('--mdrop', type=float, default=0.2, help='drop for cora middle layers')
    parser.add_argument('--drop', type=float, default=0.3, help='drop for cora layers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init cora learning rate')
    parser.add_argument('--max_length', type=int, default=15, help='the max length of chromosome')
    parser.add_argument('--max_layers', type=int, default=15, help='the max length of chromosome')
    parser.add_argument('--max_arch_length', type=int, default=15, help='the max length of chromosome')
    parser.add_argument('--min_layers', type=int, default=3, help='the min length of chromosome(startlength)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs 300')
    parser.add_argument('--init_max_len', type=int, default=15, help='the max length of chromosome')
    parser.add_argument('--num_class', type=int, default=7, help='classes of graph dataset')
    parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
    parser.add_argument('--dim', type=int, default=1, help='Dimension of B-spline')
    parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size of B-spline')
    parser.add_argument('--init_size', type=int, default=20, help='initial neighbor input size')
    parser.add_argument('--pop_size', type=int, default=20, help='initial population size(30)')
    parser.add_argument('--max_gen', type=int, default=20, help='evolution termination generation(30)')
    parser.add_argument('--prob_crossover', type=float, default=0.8, help='crossover probability')
    parser.add_argument('--prob_mutation', type=float, default=0.2, help='mutation probability')
    parser.add_argument('--mutation_types_prob', type=list, default=[0.25, 0.25, 0.25, 0.25], help='the probability of different mutation operation')
    parser.add_argument('--mutation_prob_list', type=list, default=[0.25, 0.25, 0.25, 0.25], help='probability list for mutation types')  
    parser.add_argument('--max_mutations', type=int, default=5, help='maximum number of mutations') 


    return parser.parse_args()


def get_data_and_population(args):
    print("Here are our exp settings:\n", args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)

    # Load and preprocess data
    # cora /pubmed  
    adj, features, labels, idx_train, idx_val, idx_test, edge_index, edge_attr = load_data(path="your_datapath", dataset="cora")
    # corafull
    # adj, features, labels, idx_train, idx_val, idx_test, edge_index, edge_attr = load_data()
    # adj = adj.float().to(device)

    # citerseer
    # adj, features, labels, idx_train, idx_val, idx_test, edge_index, edge_attr = load_data(dataset="citeseer")
    
    adj = aug_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None
    data = adj, features, labels, edge_index, edge_attr
    index = idx_train.to(device), idx_val.to(device), idx_test.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    # Initialize population
    population = PopulationGNN(args, adj, edge_index, edge_attr)
    init_pops = population.initialize_individuals()
    init_pops, best_model_path_f1, best_individual_f1, best_gen = train_and_eval(
        args, init_pops, data, index, 1, 0, 1)  # Use fewer epochs during BO optimization
        # args, init_pops, data, index, 1, 0, 1, epochs=50)  # Use fewer epochs during BO optimization

    pops = init_pops

    return device, data, index, population, pops, best_individual_f1, best_gen


# selected best global HPSet
global_best_params = None

def evolution_process(args, device, data, index, population, pops, best_individual_f1):
    # N_current = args.N_initial_exploitation
    exploration_phase = True
    gen_now = 1
    max_gen = args.max_gen
    best_gen = 1
    N_initial_exploration = 10
    N_initial_exploitation = 2
    N_current = N_initial_exploration
    Delta_fitness = 0.01
    Delta_N = 2
    exploration_phase = True
    phase_threshold = 0.01  # determine exploration  or  exploitation 
    patience = 5  # calculate how many times Δfitness then change stage
    all_test_results = []
    stable_counter = 0  
    fitness_improvements = []

    for curr_gen in range(1, max_gen + 1):
        avg_fitness_improvement = calculate_avg_fitness_improvement(pops)
        fitness_improvements.append(avg_fitness_improvement)
        smoothed_fitness_improvement = ewma(fitness_improvements)

        if exploration_phase:
            k = 2
            prob_crossover = 0.8
            prob_mutation = 0.2
            if smoothed_fitness_improvement < phase_threshold:
                stable_counter += 1
                if stable_counter >= patience:
                    exploration_phase = False
                    stable_counter = 0
            else:
                stable_counter = 0
        else:
            k = 3
            prob_crossover = 0.4
            prob_mutation = 0.6
            if smoothed_fitness_improvement >= phase_threshold:
                stable_counter += 1
                if stable_counter >= patience:
                    exploration_phase = True
                    stable_counter = 0
            else:
                stable_counter = 0
        
        print('NOW This is evolution!!!!!!!!!')
        
        print(f' NOW is the  {curr_gen}th!!!!!!')

        crossover_and_mutation = CrossoverAndMutation(pops, args, N_current, k, exploration_phase, prob_crossover, prob_mutation)
        through_mutatation_offs = crossover_and_mutation.process()
        through_mutatation_offs = population.create_from_offspring(through_mutatation_offs)
        print(' THE 1th of train_and_eval START !!!!!!!!!')
        through_mutatation_offs, _, _, best_gen = train_and_eval(
            args, through_mutatation_offs, data, index, curr_gen,
            1, best_gen)
        print('THE 1th of train_and_eval END !!!!!!!!!')
        print(f"Generation {curr_gen}: all_test_results_1: {all_test_results}")
        new_pops = environment_selection(pops, through_mutatation_offs, args)
        new_pops = population.create_from_offspring(new_pops)
        print('HE 2th of train_and_eval START !!!!!!!!!')
        pops, _, best_individual_f1, best_gen = train_and_eval(
            args, new_pops, data, index, curr_gen + 1,
            0, best_gen)
        print('THE 1th of train_and_eval END !!!!!!!!!')
     # EWMA calculate 
        if smoothed_fitness_improvement < Delta_fitness:
            N_current = min(N_current + Delta_N, N_initial_exploitation)
        else:
            N_current = max(N_current - Delta_N, N_initial_exploration)

        print(f"Testing the best individual of generation {curr_gen}")
        model_f1 = ModelOp(best_individual_f1, data[0], data[1].shape[1], args.hid_dim, len(torch.unique(data[2])), args.fdrop, args.mdrop, args.drop, args.dim, args.kernel_size, data[3], data[4], device=device)
        model_f1 = model_f1.to(device)


        print(f' the test  {curr_gen}th  start !!!!!!')
        test_end(
            model_f1, {'x': data[1].to(device), 'y': data[2].to(device), 'train_mask': index[0].to(device), 'val_mask': index[1].to(device),
            'test_mask': index[2].to(device)}, best_individual_f1, curr_gen, all_test_results)
    
        print(f' the test  {curr_gen}th  end!!!!!!')
        print(f"Generation {curr_gen}: all_test_results_3: {all_test_results}")

        print("*********start BO process!")
        if curr_gen % 5== 0:  # Perform BO optimization every 5 generations
            study = optuna.create_study(direction='maximize')
            objective = create_objective(data, index, population, pops, curr_gen)
            study.optimize(objective, n_trials=10, timeout=360) 
            best_params = study.best_params
            apply_best_params(args, best_params)


        print(f'evo{curr_gen}  th end!!!!!!')

        print("*********end BO process!")

        print(f' the {curr_gen}th of evolution process end!!!!!!')

        # gen_now += 1

    print("start final test stage!")
    print(f"test stage: all_test_results_4: {all_test_results}")

    best_overall_result = max(all_test_results, key=lambda x: x[2])
    print(f"Best Overall Test Accuracy: {best_overall_result[1]:.4f}, Test F1 Score: {best_overall_result[2]:.4f}, Architecture: {best_overall_result[3]}")

    best_results = sorted(all_test_results, key=lambda x: x[2], reverse=True)[:10]
    
    mean_test_acc = np.mean([result[1] for result in best_results])
    mean_test_acc_std = np.std([result[1] for result in best_results])
    # mean_test_f1 = np.mean([result[2].cpu().numpy() if isinstance(result[2], torch.Tensor) else result[2] for result in best_results])
    mean_test_f1 = np.mean([result[2] for result in best_results])
    mean_test_f1_std = np.mean([result[2] for result in best_results])

    # mean_test_f1_std = np.std([result[2].cpu().numpy() if isinstance(result[2], torch.Tensor) else result[2] for result in best_results])

    print("Best test results:")
    best_f1_score = 0
    gen_now = 1
    for result in best_results:
        print(f"Generation {result[0]}: Test Accuracy: {result[1]:.4f}, Test F1 Score: {result[2]:.4f}, Architecture: {result[3]}")
        if result[2] > best_f1_score:
            best_gen = gen_now
        gen_now = gen_now + 1
    print(f"Mean Best Test Accuracy: {mean_test_acc:.4f} ± {mean_test_acc_std:.4f}")
    print(f"Mean Best Test F1 Score: {mean_test_f1:.4f} ± {mean_test_f1_std:.4f}")

    print(f'Generations to Best Model: {best_gen}')
    print("Best Hyperparameters: ", global_best_params)
