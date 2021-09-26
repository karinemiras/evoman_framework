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
import csv

sys.path.insert(0, 'evoman') 
os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import Environment
from demo_controller import player_controller

from argparse import ArgumentParser

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

def init_population(pop_size, n_hidden, n_input, n_output):
    genotype_len = (n_input + 1) * n_hidden + (n_hidden + 1) * n_output
    pop = np.random.uniform(lower_bound, upper_bound, size=(pop_size, genotype_len))
    return np.array(pop)

# Return the winner of a k-sized random tournament with replacement based on fitness.
# This function is used for parent selection.
def tournament_selection(pop, fitnesses, k=3):
    random_indices = np.random.randint(0, len(pop), k)
    winner_idx = np.argmax(fitnesses[random_indices])
    return pop[random_indices[winner_idx]]

# crossover two parents to create two children
def crossover(parent_1, parent_2, crossover_rate):
    # randomly select whether to perform crossover
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent_1))
        off_spring_1= np.append(parent_1[:crossover_point], parent_2[crossover_point:])
        off_spring_2 = np.append(parent_2[:crossover_point], parent_1[crossover_point:])
    else:
        off_spring_1, off_spring_2 = parent_1.copy(), parent_2.copy()
    
    return [off_spring_1, off_spring_2]

# Adds gaussian noise to each allele with mutation_rate probability.
# To do: change step size to one often used in literature
def mutation(genotype, mutation_rate):
    for i in range(len(genotype)):
        if np.random.rand() < mutation_rate:
            genotype[i] += np.random.normal(0, 1)
    return genotype

def save_scores(scores, filepath):
    # open the file in the append mode
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(scores)
        f.close()



def genetic_algorithm(n_iter, n_pop, cross_rate, mutation_rate, results_path):
    # Initialize population
    pop = init_population(n_pop, n_hidden=N_HIDDEN_NEURONS, n_input=20, n_output=5)
    
    # Keep best solution
    best_solution, best_fitness = 0, evaluate(pop[0])

    # Enumerate generations
    for gen in range(n_iter):

        # Evaluate population
        fitnesses = np.array([evaluate(individual) for individual in pop])

        # Save plot results
        gen_mean = np.mean(fitnesses)
        gen_max = np.max(fitnesses) 
        save_scores((gen_mean, gen_max), results_path)
        
        # Find new optimal solution
        for i in range(n_pop):
            if fitnesses[i] > best_fitness:
                best_solution, best_fitness = pop[i], fitnesses[i]
                print("Generation {}, new best fitness = {:.3f}".format(gen+1, fitnesses[i]))
       
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
                c = mutation(c, mutation_rate)
                # store for next generation
                children.append(c)		
        # replace population
        pop = np.array(children)

    return [best_solution, best_fitness]


# define the total iterations
n_generations = 10
# define the population size
n_pop = 30
# crossover rate
crossover_r = 0.9
# mutation rate
mutation_r = 0.1
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

            # Find and save best individual
            best_solution, best_fitness = genetic_algorithm(n_generations, n_pop, crossover_r, mutation_r, results_path)
            print('Enemy {} - Generation {} finished.'.format(enemy, i+1))
            print('Best fitness = ', best_fitness)
            solution_path = os.path.join(log_path, 'solution.npy')
            np.save(solution_path, best_solution)
