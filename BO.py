"""
the whole process of PB-HBO
1. HP space
2. TPE+EI
"""

import torch
import torch.utils
import torch.nn.functional as F
from utils import *
from dataset_load import *
from fitness import *
from GA_operations import * 

def ewma(data, alpha=0.3):
    """ Exponential Weighted Moving Average (EWMA) """
    avg = data[0]
    for value in data[1:]:
        avg = alpha * value + (1 - alpha) * avg
    return avg

def create_space(data, index,population,pops,curr_gen):
    return [
        Integer(4, 256, name='hid_dim'),
        Real(0.4, 0.6, name='fdropout'),
        Real(0.2, 0.4, name='mdropout'),
        Real(0.3, 0.5, name='dropout'),
        Real(1e-4, 1e-1, "log-uniform", name='learning_rate'),
        Real(1e-5, 1e-2, "log-uniform", name='weight_decay')
    ]

# selected best global HPSet
global_best_params = None

def apply_best_params(args, best_params):
    args.hid_dim = best_params.get('hid_dim', args.hid_dim)
    args.fdrop = best_params.get('fdrop', args.fdrop)
    args.mdrop = best_params.get('mdrop', args.mdrop)
    args.drop = best_params.get('drop', args.drop)
    args.learning_rate = best_params.get('learning_rate', args.learning_rate)
    args.weight_decay = best_params.get('weight_decay', args.weight_decay)

    print("Applied Best Hyperparameters: ", best_params)


def obj_for_BO(args, data, index, population, pops, curr_gen):
    exploration_phase = True
    k = 2
    N_current = 10
    prob_crossover = 0.8
    prob_mutation = 0.2

    print('Start obj_for_BO NOW !!!!!!')
    crossover_and_mutation = CrossoverAndMutation(pops, args, N_current, k, exploration_phase, prob_crossover,prob_mutation)
    through_mutatation_offs = crossover_and_mutation.process()
    through_mutatation_offs = population.create_from_offspring(through_mutatation_offs)
    print('3TH (obj) train_and_eval START NOW !!!!!!')
    
    through_mutatation_offs, _, _, _ = train_and_eval(
        args, through_mutatation_offs, data, index, curr_gen, 1, 1)  # Use fewer epochs during BO optimization
        # args, through_mutatation_offs, data, index, curr_gen, 1, 1, epochs=200)  # Use fewer epochs during BO optimization
    print('3TH (obj) train_and_eval END NOW !!!!!!')

    new_pops = environment_selection(pops, through_mutatation_offs, args)
    new_pops = population.create_from_offspring(new_pops)
    print('4 TH (obj) train_and_eval START NOW !!!!!!')
    _, _, best_individual_f1, _ = train_and_eval(
        args, new_pops, data, index, curr_gen + 1,
        0, 1)  # Use fewer epochs during BO optimization
        # 0, 1, epochs=50)  # Use fewer epochs during BO optimization
    print('4TH (obj) train_and_eval END NOW !!!!!!')

    print('END obj_for_BO NOW !!!!!!')
    return best_individual_f1


def create_objective( data, index,population,pops,curr_gen):

    def objective(trial):
        args = setup_args()
        params = {
            'hid_dim': trial.suggest_int('hid_dim', 4, 256),
            'fdrop': trial.suggest_float('fdrop', 0.4, 0.6),
            'mdrop': trial.suggest_float('mdrop', 0.2, 0.4),
            'drop': trial.suggest_float('drop', 0.3, 0.5),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        }
        apply_best_params(args, params)

        best_individual_f1 = obj_for_BO(args, data, index, population, pops, curr_gen)
        print('best_individual_f1[f1]:',best_individual_f1['f1'])
        return best_individual_f1['f1'].item()  # Minimize negative accuracy
    return objective