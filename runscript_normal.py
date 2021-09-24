################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller_normal import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from bio_dingen import crossover, mutation, get_children, fitfunc
import csv


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for enemy in [1,4,5,6,7,8]:
    for fitter in ["standard", "exponential", "oscilation", "errfoscilation"]:
        n_hidden_neurons = 10       #number of hidden neurons
        enemy = int(enemy)          #which enemy
        run_nr = 2                  #number of runs
        generations = 100           #number of generations per run
        population_size = 50       #pop size
        mutation_baseline = 0.05    #minimal chance for a mutation event
        mutation_multiplier = 0.2   #fitness dependent multiplier of mutation chance
        repeats = 5
    
        experiment_name = f'enemy_{enemy}'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
    
    
        for run in range(run_nr):
    
            # initializes simulation in individual evolution mode, for single static enemy.
            env = Environment(experiment_name=experiment_name,
                              enemies=[enemy],
                              playermode="ai",
                              player_controller=player_controller(n_hidden_neurons),
                              enemymode="static",
                              level=2,
                              speed="fastest",
    			  randomini = "yes")
    
            # number of weights for multilayer with 10 hidden neurons
            n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    
            #initiate 100 parents
            pop = np.random.uniform(-1, 1, (population_size ,n_vars))
            total_fitness_data = []
            children_index = []
            children_data = []
            max_health = 0
            best = []
    
            for g in range(generations):
                if max_health > 99.9:
                    break
                fitness_array = []
                fitness_array_smop = []
                
                for player in pop:
                    fitness_new = 0
                    fitness_smop = 0
                    health = 0
                    
                    for i in range(repeats):
                        f, p, e, t = env.play(pcont=player)
                        fitness_new += (0.9*(100 - e) + 0.1*p - np.log(t))*(1/repeats)
                        fitness_smop += fitfunc(fitter, generations, g, t, e, p)*(1/repeats)
                        health += (1/repeats)*p
                        
                    fitness_array.append(fitness_new)
                    fitness_array_smop.append(fitness_smop)
        
                    #save the children data (big file)
                    children_index.append([g, f, p, e, t])
                    children_data.append(player)
    
                    #save the maximum achieved health
                    if health > max_health and e == 0:
                        max_health = health
                        best = player
        
                    #save the fitness data
                    total_fitness_data.append([np.max(fitness_array),
                                               np.mean(fitness_array),
                                               np.std(fitness_array)])
        
                pop = get_children(pop, np.array(fitness_array_smop),
                                   mutation_baseline, mutation_multiplier)
            
                print(f'Run: {run}, Fitter: {fitter}, Generation {g}, fitness_best/mean = {round(np.max(fitness_array),2)} and {round(np.mean(fitness_array),2)},, fitness_smop best/mean = {round(np.max(fitness_array_smop),2)} and {round(np.mean(fitness_array_smop),2)},, best_avg_health = {round(max_health,2)}')
        
        with open(f'data_normal/{experiment_name}/fitness_data_{run}_{fitter}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([enemy, generations, max_health])
            writer.writerows(total_fitness_data)
            
        children_data = np.array(children_data)
        # with open(f'data_normal/{experiment_name}/full_data_index_{run}.csv', 'w', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['generation', 'fitness', 'p_health', 
        #                      'e_health', 'time'])
        #     writer.writerows(children_index)
            
        with open(f'data_normal/{experiment_name}/full_data_{run}_{fitter}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(children_data)
        
        with open(f'data_normal/{experiment_name}/best_sol_{run}_{fitter}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(best)
