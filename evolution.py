##################################################################################
# This file could contain the main methods for doing simulation, evolution, 
# the genetic algorithm, etc.
##################################################################################

import numpy as np

def initialize_generation(population_size, num_genes):
	"""
	Randomly initializes a generation.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	return np.random.uniform(-1, 1, (population_size, num_genes))


def generate_next_generation(population):
	"""
	Generates next generation from current population.
	:param population_fitness: array containing each individual in population and their fitness score
	"""
	return

def reproduce(x, y):
	"""
	Generate one offspring from individuals a and b using crossover and additional random mutation.
	:param x: first individual ; numpy vector with shape (num_genes,) corresponding to weights in NN controller
	:param y: second individual ; numpy vector with shape (num_genes,) corresponding to weights in NN controller
	"""
	return

def compute_fitness(env, x):
	"""
	Evaluate the fitness of individual x.
	:param x: individual x
	"""
	fitness, player_life, enemy_life, time = env.play(pcont=x)
	return fitness