from search_space import *
from GA_operations import *
from utils import *
from population import *
from fitness import *
from dataset_load import *
from BO import *
from evolution import *

if __name__ == "__main__":
    args = setup_args()
    print('START get_data!!!!!')
    device, data, index, population, pops, best_individual_f1, best_gen = get_data_and_population(args)
    print('START gp_minimize !!!!!')
    curr_gen = 1 
    
    study = optuna.create_study(direction='maximize')
    objective = create_objective(data, index, population, pops, curr_gen)
    study.optimize(objective, n_trials=50, timeout=360)
    print('END TPE!!!!!')

    best_params = study.best_params
    print('START apply_best !!!!')

    # apply_best_params(args, dict(zip([dim.name for dim in space], best_params)))
    apply_best_params(args, best_params)
    print('START evolution stage!')
    
    evolution_process(args, device, data, index, population, pops, best_individual_f1)