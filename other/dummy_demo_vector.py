################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from numpy.core.numeric import cross
sys.path.insert(0, 'evoman') 
from environment import Environment

# other imports
import numpy as np
import matplotlib.pyplot as plt

# change name to corresponding experiment run; stores a single run (?)
experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

"""
IMPLEMENT:
    - Genotype
    - Population
    - Phenotype
    - Fitness Function
    - Selection
    - Mutation
    - Crossover
"""
                                ### CONSTANTS ###

DEBUG = True # enables debug/print of certain variables

                              ### HYPERPARAMETERS ###

n_pop = 100 # population size
n_vals = 5 # values in chromosome
n_gen = 15 # number of generations
r_cross = 0.8 # crossover rate
r_mutate = 1/n_vals # mutation rate; 1/m where m is the amount of values in the chromosome

# Activation Functions
def sigmoid(x):
    # returns values between 0 & 1
    return 1./(1.+np.exp(-x))

def tanh(x):
    # returns values between -1 & 1; penalize certain moves via negative values?
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))




"""
# Enumerate generations
for gen in range(n_gen):
    # EVALUATE
    scores = [fitness(c) for c in population] # evaluate fitness of candidates in population
#TODO: IMPLEMENT FITNESS FUNCTION
"""

def default_fitness(x):
    # takes in array of values & returns their sum; maximize sum
    return sum(x)


# TOURNAMENT SELECTION
def tournament_selection(population, scores, k=3):
    # randomly select candidate from population
    selection_candidate = np.random.randint(len(population))
    for candidate in np.random.randint(0, len(population), k-1):
        # perform tournament selection by comparing candidate scores
        if scores[candidate] < scores[selection_candidate]:
            selection_candidate = candidate
    return population[selection_candidate]




# CROSSOVER
def crossover(parent_1, parent_2, r_cross):
    # copy parents into children; don't want to mess up original vals
    child_1, child_2 = parent_1.copy(), parent_2.copy()
    # check for probability of recombination
    if np.random.rand() < r_cross:
        # select crossover point
        crossover_point = np.random.randint(1, len(parent_1)-2) # with len = 5, crossover point will be between idx 1:3
        # perform crossover
        child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
        child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return [child_1, child_2]


# MUTATION
# TODO: fix mutation operator - multiplying by random uniform value only keeps decreasing value; no good new solutions found
def mutation(chromosome, r_mutate):
    for i in range(len(chromosome)):
        if np.random.rand() < r_mutate:
            # perturb value by random value between 0,1 drawn from a uniform distribution
            chromosome[i] = np.random.uniform(0,1) * chromosome[i]
    



# GENETIC ALGORITHM
def genetic_algorithm(fitness_function, n_vals, n_gen, n_pop, r_cross, r_mutate, DEBUG):
    # create initial population with random floating values
    population = [np.random.uniform(0,1, n_vals).tolist() for _ in range(n_pop)]
    if DEBUG: print(population)
    # store best solutions
    best, best_fitness = 0, fitness_function(population[0])
    # iterate through generations
    for gen in range(n_gen):
        print("In generation %d" % gen)
        # evaluate all candidates in the population
        scores = [fitness_function(candidate) for candidate in population]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_fitness:
                # assign chromosome & its fitness to best & best_fitness if larger than current best
                best, best_fitness = population[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, population[i], scores[i]))
        # select parents
        selected = [tournament_selection(population, scores) for _ in range(n_pop)]
        # create next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get pair of parents
            parent_1, parent_2 = selected[i], selected[i+1]
            for candidate in crossover(parent_1, parent_2, r_cross):
                mutation(candidate, r_mutate)
                # store children
                children.append(candidate)
        # replace population with next generation
        population = children
    return [best, best_fitness]
    

results = genetic_algorithm(default_fitness, n_vals, n_gen, n_pop, r_cross, r_mutate, DEBUG)
print (results)


# initializes environment with ai player using random controller, playing against static enemy
#env = Environment(experiment_name=experiment_name)
#env.play()

