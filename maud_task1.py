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
import pickle
import csv
import pandas as pd

class specialist:
	def __init__(self,neurons=20,gen=50,popsize=50,pc=0.3,pm=0.8,EA='gaus',enemy=2,subname='exp1'):

		self.gen = gen # Number of generations
		self.pc = pc # Crossover probability
		self.pm = pm # Mutation probability
		self.EA = EA
		self.enemy = enemy

		# Make a folder for results
		self.experiment_name = 'EA' + self.EA + '/enemy%i'%self.enemy
		if not os.path.exists(self.experiment_name):
		    os.makedirs(self.experiment_name)

		fitness_results  = open(self.experiment_name + subname + '.txt','a')

		self.env = Environment(experiment_name=self.experiment_name,
						       enemies=[5],
						       level=2,
						       playermode='ai',
						       enemymode='static',
						       player_controller=player_controller(neurons),
						       speed='fastest')

		n_vars = (self.env.get_num_sensors()+1)*neurons + (neurons+1)*5

		# Create framework for individuals/population
		creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # What do these weights mean?
		creator.create("Individual", list, fitness=creator.FitnessMax)

		self.toolbox = base.Toolbox()
		self.toolbox.register("new_ind", random.uniform,-1,1)
		self.toolbox.register("individual", tools.initRepeat, creator.Individual,
		                 self.toolbox.new_ind, n=n_vars)

		self.toolbox.register("select", tools.selTournament, tournsize=3)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		self.toolbox.register("mate", tools.cxBlend,alpha=0.5)
		self.toolbox.register("evaluate", self.evaluate)

		if self.EA == 'gaus':
			self.toolbox.register("mutate", tools.mutGaussian,mu=0,sigma=1,indpb=0.2)
		else:
			self.toolbox.register("mutate", tools.mutUniformInt,low=1,up=1,indpb=0.2)


		self.pop = self.toolbox.population(n=popsize)

		# Set up some tools for calculating statistics
		self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
		self.stats.register("avg", np.mean, axis=0)
		self.stats.register("std", np.std, axis=0)
		self.stats.register("min", np.min, axis=0)
		self.stats.register("max", np.max, axis=0)

		# Calculate the fitness for every member in the new population
		fitnesses = np.array(list(map(lambda y: self.toolbox.evaluate(self.env,y), self.pop)))
		
		for ind, fit in zip(self.pop, fitnesses):
			ind.fitness.values = [fit]

		self.record = self.stats.compile(self.pop)

		# Save the best individual --> for later use
		self.champion = {'fitness': -100, 'ind': []}

		for ind in self.pop:
			if ind.fitness.values[0] > self.champion['fitness']:
				self.champion['fitness'] = ind.fitness.values
				self.champion['ind'] = ind

	def evaluate(self,env,ind):
		""" Play the game and calculate resulting fitness. """
		fitness,_,_,_ = env.play(pcont=np.array(ind))
		return fitness

	def cycle(self):
		""" Evolutionary algorithm. """

		for c in range(self.gen):
			print("Gen:", c,len(self.pop))

			# Parent selection using Tournament selection
			offspring = self.toolbox.select(self.pop,4)
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

			# Calculate the statistic of population in new generation
			for keys in self.record.keys():
				new_record = self.stats.compile(self.pop)

				self.record[keys] = np.append(self.record[keys],new_record[keys])

			# Check if found better solution
			for ind in self.pop:
				if ind.fitness.values[0] > self.champion['fitness']:
					self.champion['fitness'] = ind.fitness.values
					self.champion['ind'] = ind

		# Save results as a pickle
		pickle.dump(self.record, open(self.experiment_name+subname + ".p","wb"))
		
		# Save the final best solution
		np.save(str(self.experiment_name+self.subname+'_bestsolution'), self.champion['ind'])


experiments = [1,2,3,4,5] #ENTER WHICH EXPERIMENT NUMBERS YOU ARE RUNNING
ea = 'gaus' #ENTER WHICH EA YOU ARE RUNNING ('gaus' or 'uni')
en = 2 #ENTER WHICH ENEMY YOU ARE RUNNING (2,5 OR 6)

# for i in range(0,len(experiments)):
# 	random.seed(experiments[i])
#     test = specialist(neurons=20,gen=50,popsize=50,pc=0.3,pm=0.8,EA=ea,enemy=en,subname='/exp%i'%experiments[i])
#     test.cycle()

for i in range(0,len(experiments)):
	random.seed(experiments[i])
	test = specialist(neurons=1,gen=2,popsize=5,pc=0.3,pm=0.8,EA=ea,enemy=en,subname='/exp%i'%experiments[i])
	test.cycle()
