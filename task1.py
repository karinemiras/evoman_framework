################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from deap import base
from deap import tools
import numpy as np
import random
from deap import creator
from demo_controller import player_controller


class specialist:
	def __init__(self,neurons=10,gen=10,popsize=5,pc=0.5,pm=0.5):

		self.gen = gen # Number of generations
		self.pc = pc # Crossover probability
		self.pm = pm # Mutation probability

		# Make a folder for results
		experiment_name = 'task1_1'
		if not os.path.exists(experiment_name):
		    os.makedirs(experiment_name)

		self.env = Environment(experiment_name=experiment_name,
						       enemies=[5],
						       level=2,
						       playermode='ai',
						       enemymode='static',
						       player_controller=player_controller(neurons),
						       speed='fastest')

		n_vars = (self.env.get_num_sensors()+1)*neurons + (neurons+1)*5

		# Create framework for individuals/population
		creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0)) # What do these weights mean?
		creator.create("Individual", list, fitness=creator.FitnessMax)

		self.toolbox = base.Toolbox()
		self.toolbox.register("new_ind", np.random.uniform,-1,1)
		self.toolbox.register("individual", tools.initRepeat, creator.Individual,
		                 self.toolbox.new_ind, n=n_vars)

		self.toolbox.register("select", tools.selTournament, tournsize=2)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		self.toolbox.register("mate", tools.cxTwoPoint)
		self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
		self.toolbox.register("evaluate", self.evaluate)

		self.pop = self.toolbox.population(n=popsize)

	def evaluate(self,env,ind):
		""" Play the game and calculate resulting fitness. """
		fitness,_,_,_ = env.play(pcont=np.array(ind))
		return fitness

	def cycle(self):
		""" Evolutionary algorithm. """

		for c in range(self.gen):
			print("Gen:", c,len(self.pop))

			# Parent selection using Tournament selection
			offspring = self.toolbox.select(self.pop, len(self.pop))
			offspring = list(map(self.toolbox.clone, offspring))
			
			 # 1-point crossover on offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
			    if random.random() < self.pc:
			        self.toolbox.mate(child1, child2)
			        del child1.fitness.values
			        del child2.fitness.values

			# Apply mutation on the offspring
			for mutant in offspring:
			    if random.random() < self.pm:
			        self.toolbox.mutate(mutant)
			        del mutant.fitness.values

			# Calculate the fitness for the new offspring
			fitnesses = np.array(list(map(lambda y: self.toolbox.evaluate(self.env,y), offspring)))
			
			for ind, fit in zip(offspring, fitnesses):
				ind.fitness.values = [fit]

			print('f:',fitnesses)

			# The population is entirely replaced by the offspring
			# self.pop[:] = offspring

			# Or: Survival of the fittest
			newpop = self.pop + offspring
			self.pop = self.toolbox.select(newpop, len(self.pop))

test = specialist(neurons=10,gen=20,popsize=20,pc=1,pm=0.2)
test.cycle()