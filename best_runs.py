"""
Created on Wed Sep 22 12:35 2021

@author: iris
make data for boxplots
"""
# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller_memory import player_controller

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


experiment_name = 'data_memory/enemy_1_standard/boxplots'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10       #number of hidden neurons
enemy = 1               #which enemy TODO
run_nr = 10                 #number of runs
population_size = 100       #pop size
test_number = 5             #times to run the best individual
folder = 'data_memory/enemy_1_standard'         #folder where the best individuals are saved

for run in range(run_nr):
    weights_data = []
    total_fitness_data = []

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      randomini = "yes")

    #open data
    with open(f'{folder}/best_sol_{run}.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            weights_data.append(row)
        weights = np.array(weights_data[0])

    #test the specific individual multiple times
    f_list, p_list, e_list, t_list = [], [], [], []
    for r in range(test_number):
        f, p, e, t = env.play(pcont=weights)
        f_list.append(f)
        p_list.append(p)
        e_list.append(e)
        t_list.append(t)

    #save the fitness data
    total_fitness_data.append(["fitness", "p_health", "e_health", "time"])
    total_fitness_data.append([np.mean(f_list),
                               np.mean(p_list),
                               np.mean(e_list),
                               np.mean(t_list)])

    with open(f'{experiment_name}/fitness_data_{run}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(total_fitness_data)
