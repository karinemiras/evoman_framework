################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from bio_dingen import crossover, mutation, get_children
import csv


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
enemy = 5

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

#initiate 100 parents
population_size = 100
pop = np.random.uniform(0, 1, (population_size ,n_vars))
total_fitness_data = []

generations = 20
for g in range(generations):
    print(f'#{g}#')
    fitness_array = []
    for player in pop:
        f, p, e, t = env.play(pcont=player)
        fitness_array.append(f)
    
    #save the fitness data
    total_fitness_data.append([np.max(fitness_array), 
                               np.mean(fitness_array), 
                               np.std(fitness_array)])
    
    
    pop = get_children(pop, np.array(fitness_array))
    

with open('testing_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([enemy, generations, population_size])
    writer.writerows(total_fitness_data)
