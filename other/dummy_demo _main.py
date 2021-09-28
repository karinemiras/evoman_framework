################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
# imports general
import numpy as np
import matplotlib.pyplot as plt

# imports framework
import sys, os
from numpy.core.numeric import cross
sys.path.insert(0, 'evoman') 
from environment import Environment


# change name to corresponding experiment run; stores a single run (?)
experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initialize environment
env = Environment(experiment_name=experiment_name)


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

n_pop = 10 # population size - amount of genomes
genome_length = 5 * (env.get_num_sensors()+1)
n_vals = genome_length # values in chromosome
n_gen = 5 # number of generations
r_cross = 0.8 # crossover rate
r_mutate = 1/n_vals # mutation rate; 1/m where m is the amount of values in the chromosome

# Activation Functions
def sigmoid(x):
    # returns values between 0 & 1
    return 1./(1.+np.exp(-x))

def tanh(x):
    # returns values between -1 & 1; penalize certain moves via negative values?
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# FITNESS FUNCTION
def default_fitness(x):
    # takes in array of values & returns their sum; maximize sum
    return sum(x)

def fitness_float64(x):
    total = 0
    for i in np.nditer(x):
        total += i 
    return total

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
def crossover(parent_1, parent_2, r_cross, DEBUG):
    # copy parents into children; don't want to mess up original vals
    child_1, child_2 = parent_1.copy(), parent_2.copy()
    if DEBUG: print(f'Parent 1: {parent_1, len(parent_1)}, \n Parent 2: {parent_2, len(parent_2)}')
    # check for probability of recombination
    if np.random.rand() < r_cross:
        # select crossover point
        crossover_point = np.random.randint(1, len(parent_1)-2)
        if DEBUG: print(f'CROSSOVER POINT: {crossover_point}')
        # perform crossover
        if DEBUG: print(f'Parent 1 CP: {parent_1[:crossover_point]}, Parent 2 CP: {parent_2[crossover_point:]}')
        # child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
        # child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
        child_1 = np.append(parent_1[:crossover_point], parent_2[crossover_point:])
        child_2 = np.append(parent_2[:crossover_point], parent_1[crossover_point:])
    return [child_1, child_2]


# MUTATION
# TODO: fix mutation operator - multiplying by random uniform value only keeps decreasing value; no good new solutions found
def mutation(chromosome, r_mutate):
    for i in range(len(chromosome)):
        if np.random.rand() < r_mutate:
            # perturb value by random value between 0,1 drawn from a uniform distribution
            chromosome[i] = np.random.uniform(0,1) * chromosome[i]

# TODO: fix mutation operator... is there something else going wrong? Seemingly no new solutions found i.e. no mutation happening
def polynomial_mutation(chromosome, r_mutate, DEBUG):
    for i in range(len(chromosome)):
        # iterate through values of chromosome
        index_parameter = i
        delta_L = (2 * np.random.rand() ** 1 / (1 + index_parameter) - 1)
        if np.random.rand() < r_mutate:
            # perturb value
            if DEBUG: print(f'Chromosome Value Old: {chromosome[i]}')
            if DEBUG: print(f'CHROMOSOME MUTATION: {chromosome + delta_L * (chromosome - chromosome[i])}') 
            chromosome = chromosome + delta_L * (chromosome - chromosome[i])
            if DEBUG: print(f'Chromosome Value New: {chromosome[i]}')

def swap_mutate(chromosome, r_mutate, DEBUG):
    for i in range(len(chromosome)):
        if np.random.rand() < r_mutate:
            generated_index = 0
            while True:
                generated_index = np.random.randint(0, len(chromosome)-1)
                if (generated_index != i): break
            # swap values
            if DEBUG: print(f'Swap {generated_index} with {i}')
            chromosome[i], chromosome[generated_index] = chromosome[generated_index], chromosome[i]


# GENETIC ALGORITHM
def genetic_algorithm(fitness_function, mutation_function, n_vals, n_gen, n_pop, r_cross, r_mutate, DEBUG):
    # create initial population
    population = []
    for i in range(n_pop):
        genome = np.random.uniform(-1,1, size=(n_vals,))
        population.append(genome)

    #if DEBUG: print(len(population), population)
    
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
                best, best_fitness,generation_found_in = population[i], scores[i], gen
                #print(">%d, new best f(%s) = %.3f" % (gen, population[i], scores[i]))
        # select parents
        selected = [tournament_selection(population, scores) for _ in range(n_pop)]
        # create next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get pair of parents
            parent_1, parent_2 = selected[i], selected[i+1]
            for candidate in crossover(parent_1, parent_2, r_cross, DEBUG=False):
                mutation_function(candidate, r_mutate, DEBUG)
                # store children
                children.append(candidate)
        # replace population with next generation
        population = children
    return [best, best_fitness, generation_found_in]
    

results = genetic_algorithm(fitness_float64, swap_mutate, n_vals, n_gen, n_pop, r_cross, r_mutate, DEBUG)
print (results)



#env.play()

