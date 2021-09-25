################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

"""
TO DO:
https://ieeexplore.ieee.org/document/350042
useful link comparing selection mechanisms

Reason about evolution mechanisms and 
implement the theoretically optimal one.

Then, fine-tune the parameters. Try to get >85 for enemy 3.
# For grid search: use npop=30 and gens=10, compare the mean fitness
# For final scores: use npop=100 and gens=30
"""


# imports framework
import sys
import os
import time

sys.path.insert(0, 'evoman') 
os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import Environment
from demo_controller import player_controller

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

N_HIDDEN_NEURONS = 10

# This is a singleton so that I don't have to use global variables
class Experiment:
    def initialize(name, enemy):
        Experiment.env = Environment(experiment_name=name,
                                     player_controller=player_controller(N_HIDDEN_NEURONS), # Use demo_controller
                                     enemies=[enemy],
                                     level=2,
                                     randomini='yes',
                                     savelogs='yes',  # Logging done manualy   
                                     clockprec='low',
                                     playermode="ai",
                                     speed="fastest",
                                     enemymode="static",
                                     contacthurt="player")
        Experiment.best_genome = None
        Experiment.best_gain = -101

# Function to evaluate candidate solutions
def evaluate(individual):
    fitness, p, e, t = Experiment.env.play(pcont=individual)
    return fitness

#n_vars = (env.get_num_sensors()+1)*10 + (10+1)*5
def init_population(pop_size, n_hidden, n_input, n_output):
    genotype_len = (n_input + 1) * n_hidden + (n_hidden + 1) * n_output
    pop = np.random.uniform(lower_bound, upper_bound, size=(pop_size, genotype_len))

    return np.array(pop)
    #return np.vsplit(pop, pop_size) # Split into array of genotype arrays

# tournament selection (https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/)
# Compares random individual to k other individuals 
# and returns the best one.
def tournament_selection(pop, fitnesses, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if fitnesses[ix] < fitnesses[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, crossover_rate):
    # p1 = np.array(p1)
    # p2 = np.array(p2)
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < crossover_rate:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)) # originally set to -2
        # perform crossover
        c1 = np.concatenate((p1[:pt], p2[pt:]))
        c2 = np.concatenate((p2[:pt], p1[pt:]))
    return [c1, c2]

# Add gaussian noise to each allele with mutation_rate probability.
# To do: change step size to one often used in literature
def mutation(genotype, mutation_rate):
    for i in range(len(genotype)):
        if np.random.rand() < mutation_rate:
            genotype[i] += np.random.normal(0, 1)
    return genotype

import csv

def save_scores(scores, filepath):
    # open the file in the write mode
    with open(filepath, 'a', newline='', encoding='utf-8') as f:

        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(scores)

        # close the file
        f.close()



def genetic_algorithm(n_iter, n_pop, cross_rate, mutation_rate, results_path):
    # Initialize population
    pop = init_population(n_pop, n_hidden=N_HIDDEN_NEURONS, n_input=20, n_output=5)
    
    # keep track of best solution
    best, best_eval = 0, evaluate(pop[0])

    # Enumerate generations
    for gen in range(n_iter):

        # Evaluate population
        fitnesses = [evaluate(individual) for individual in pop]

        # Save plot results
        gen_mean = np.mean(fitnesses)
        gen_max = np.max(fitnesses) 
        save_scores((gen_mean, gen_max), results_path)
        
        # check for new best solution
        for i in range(n_pop):
            if fitnesses[i] > best_eval:
                best, best_eval = pop[i], fitnesses[i]
                print(">gen %d, new best fitness = %.3f" % (gen, fitnesses[i]))
        # Select parents
        selected = [tournament_selection(pop, fitnesses) for _ in range(n_pop)]
        
        # create the next generation
        children = []
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, cross_rate):
                # mutation
                mutation(c, mutation_rate)
                # store for next generation
                children.append(c)		
        # replace population
        pop = children

    return [best, best_eval]


# define the total iterations
n_generations = 2
# define the population size
n_pop = 4
# crossover rate
crossover_r = 0.9
# mutation rate
mutation_r = 0.9  
# pop initialization bounds
upper_bound = 1
lower_bound = -1

if __name__ == '__main__':
    all_gains = {}

    # For each enemy
    enemies = [3]
    n_experiments = 1
    for enemy in enemies:
        
        # Run N independent experiments
        for i in range(n_experiments):

            log_path = Path('EA2', 'enemy-{}'.format(enemy), 'run-{}'.format(i))
            log_path.mkdir(parents=True, exist_ok=True)
            Experiment.initialize(str(log_path), enemy)
            results_path = os.path.join(log_path, 'results.csv')

            # Remove previous experiment results
            if os.path.exists(results_path):
                os.remove(results_path)

            # perform the genetic algorithm search
            best, score = genetic_algorithm(n_generations, n_pop, crossover_r, mutation_r, results_path)
            print('Done!')
            print('f(%s) = %f' % (best, score))
