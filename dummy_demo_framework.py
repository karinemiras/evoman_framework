################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
# imports general
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

class Experiment:
    """
    Class used to represent the experiment.

    Attributes
    ----------
    experiment_name : name of the experiment - can be extended; e.g. increment name by 1 every time an experiment is run or add date

    Methods
    -------
    check_path : checks if a path exists for the current filename
    plot_data : takes in values of the current run (e.g. fitness, average_mean, std_dev) & plots the data
    """
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def check_path(self):
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

    def plot_data():
        """
        Uses matplotlib to visualize data collected during the experiment: not sure about the idea of "current" vs. "total" data.
        Use of parameters will depend on time when we decide to plot the data - after all generations have passed? NO - need to plot mean for each generation.
        
        Params
        ------
        n_gen : number of generations
            - hyperparams
        c_gen : current generation
            - i.e. f'gen{current iteration}' becomes label for x axis
        c_mean : mean fitness of current generation
            - c_mean = total_fitness/n_pop
        c_std_dev : standard deviation from the mean of the current generation
            - c_std_dev = np.std(population)
        c_max : max fitness value of current generation
            - c_max = best_fitness
        c_algo : name of current solution
        c_experiment : experiment name
        c_enemy : current enemy type
        (optional) t_mean : total mean of the whole evolutionary cycle (after all generations)
            - t_mean = sum c_mean of all gen
        """
        pass


class RandomValues:
    """
    Class which implements the various random methods using numpy. The idea was to instantiate RandomValues
    to have a random number generator (RNG) with a set value. Simply allows for "controlled" randomization. Plus
    we can simply say rv = RandomValues() and use rv.function() to call a specific method, rather than the more verbose
    np.random.function().

    Attributes
    ----------
    random_seed : initialize a random number generator, to enable replication

    Methods
    -------
    random_int : np.randint
    random_uniform : np.uniform
    """
    def __init__(self, random_seed):
        self.random_seed = np.random.default_rng(42)

    def random_int(x, y):
        return np.random.randint(x, y)

    def random_uniform(x, y):
        return np.random.uniform(x, y)

    def print_random_seed(self):
        print (self.random_seed)

class ActivationFunction:
    """
    Class which implements various activation functions.

    Methods
    -------
    sigmoid : return value between 0 & 1 (values towards -∞ become 0; values towards +∞ become 1)
    tanh : return value between -1 & 1 (valuable if network layers > 1)
    Gaussian : return value between 0 & 1 (probability distribution)
    """
    def sigmoid(x):
        pass

    def tanh(x):
        pass

    def Gaussian(x):
        pass

class FitnessFunction:

    def default_fitness():
        pass

    def temp_fitness(self, x):
        """Random fitness function used only for debugging. Sums values of an array x

        Params
        -----------
        x : numpy array
         """
        total = 0
        for i in np.nditer(x):
            total += 1

        return total


class Mutation:
    """
    Class which implements various mutation methods. 

    Mutation rate is handled by Genetic Algorithm; specifically create_next_gen()

    """

    def swap_mutation(self):
        """
        Generate a random index & swap the values between two indices of a child
        """
        pass

    def polynomial_mutation(self):
        """
        Perturb value of a random index by some amount
        """
        pass

class CrossOver:
    """
    Class which implements crossover methods


    """

    def crossover(self):
        pass

class Selection:
    """
    Class which implements selection methods
    
    """

    def tournament_selection(pop, scores, k):
        pass

  
### CLASS FOR GA ###
class GeneticAlgorithm:
    """Class for creating a genetic algorithm with hyperparameters specified in the constructor. Used as a blueprint
    for the GA methods specified below.

    """

    def __init__(self, population, n_pop, n_vals, n_gen, r_mutate, r_cross):
        """
        Params:
        -------
        population : empty list
        n_pop : size of the population
        n_vals : values inside the chromosome
        n_gen : number of generations
        r_mutate : mutation rate
        r_cross : crossover rate
        """
        self.population = population
        self.n_pop = n_pop
        self.n_vals = n_vals
        self.n_gen = n_gen
        self.r_mutate = r_mutate
        self.r_cross = r_cross

    def initialize_population(self):
        """
        Initializes a population of size n_pop, with n_vals values. Values are drawn from a random uniform distribution
        between -1, 1. 
        Returns: a list of size n_pop
        """
        for i in range(self.n_pop):
            genome = np.random.uniform(-1, 1, size=(self.n_vals,))
            self.population.append(genome)
        return self.population

    def create_next_gen(self, selected, cross, mutate):
        """
        Creates the next generation by picking a pair of parents and applying crossover, and randomly applying mutation

        Params
        --------
        selected : list of selected candidates from the population; selection mechanism specified in GA
        cross : crossover method specified in GA
        mutate : mutation method specified in GA

        Returns
        --------
        list of children
        """
        children = list()
        for i in range(0, self.n_pop, 2):
            # Get pair of parents
            parent_1, parent_2 = selected[i], selected[i+1]
            for candidate in cross.crossover(parent_1, parent_2, cross.r_cross):
                mutate(candidate, self.r_mutate)
                children.append(candidate)
        return children

    def generic_GA(self, DEBUG):
        """Uses basic crossover, swap mutation, and tournament selection.
        
        """
        # Instantiate functions
        fit = FitnessFunction()
        select = Selection()
        cross = CrossOver().crossover()
        mutate = Mutation().swap_mutation()

        # Initialize population
        self.population = self.initialize_population()

        best, best_fitness = 0, fit.temp_fitness(self.population[0])
        # Iterate through generations
        for gen in range(self.n_gen):
            if DEBUG: print(f"CURRENT GENERATION: {gen}")
            # Evaluate all candidates in the population by calculating fitness
            scores = [fit.temp_fitness(candidate) for candidate in self.population]
            # Check for new best by comparing each score with the current best fitness
            for i in range(self.n_pop):
                if scores[i] > best_fitness:
                    best, best_fitness = self.population[i], scores[i]
            # Select parents
            selected = [select.tournament_selection(self.population, scores) for _ in range(self.n_pop)]
            # Create next generation
            self.population = self.create_next_gen(selected, cross, mutate)

            # PLOT DATA


        return [best, best_fitness]

         


### EXAMPLE CONSTRUCTION ###
# Construct Environment
import sys, os
from numpy.core.numeric import cross
sys.path.insert(0, 'evoman')
from environment import Environment

experiment = Experiment("example_construction")
experiment.check_path()

env = Environment(experiment_name=experiment.experiment_name)

# Declare Hyperparams
population      = []
n_population    = 10
genome_length   = 5 * (env.get_num_sensors()+1)
n_values        = genome_length
n_generations   = 3
r_mutation      = 1/n_values
r_crossover     = 0.8
   
# Instantiate GA
ga = GeneticAlgorithm(population, n_population, n_values, n_generations, r_mutation, r_crossover)

# Run generic GA
ga.generic_GA(DEBUG=False)

