###############################################################################
# EvoMan FrameWork - V1.0 2016  			                      			  #
# DEMO : Neuroevolution - Genetic Algorithm with neural network.              #
# Author: Karine Miras        			                             		  #
# karine.smiras@gmail.com     				                     			  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os







class environm(Environment):

	# implements fitness function
	def fitness_single(self):

		if self.contacthurt == "player":
			return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())

		else:
			return 0.9*(100 - self.get_playerlife()) + 0.1*self.get_enemylife() - np.log(self.get_time())


experiment_name = 'co_demo'
if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation for coevolution evolution mode.
env = environm(experiment_name=experiment_name,
			   enemies=[2],
			   playermode="ai",
			   enemymode="ai",
			   player_controller=player_controller(n_hidden_neurons),
			   enemy_controller=enemy_controller(n_hidden_neurons),
			   level=2,
			   speed="fastest")


env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype/weights for perceptron phenotype network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 100
gens = 100
mutation = 0.2
last_best = 0


# runs simulation
def simulation(env,x1,x2):
	f,p,e,t = env.play(pcont=x1,econt=x2)
	return f

# normalizes
def norm(x,pfit_pop):

	if ( max(pfit_pop) - min(pfit_pop) ) > 0:
		x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
	else:
		x_norm = 0

	if x_norm <= 0:
		x_norm = 0.0000000001
	return x_norm


# evaluation
def evaluate(x1,x2):
	return np.array(list(map(lambda y: simulation(env,y,x2), x1)))


# tournament
def tournament(pop,fit_pop):
	c1 =  np.random.randint(0,pop.shape[0], 1)
	c2 =  np.random.randint(0,pop.shape[0], 1)
	if fit_pop[c1] > fit_pop[c2]:
		return pop[c1][0]
	else:
		return pop[c2][0]


# limits
def limits(x):

	if x>dom_u:
		return dom_u
	elif x<dom_l:
		return dom_l
	else:
		return x


# crossover
def crossover(pop,fit_pop):

	total_offspring = np.zeros((0,n_vars))


	for p in range(0,pop.shape[0], 2):
		p1 = tournament(pop,fit_pop)
		p2 = tournament(pop,fit_pop)

		n_offspring =   np.random.randint(1,3+1, 1)[0]
		offspring =  np.zeros( (n_offspring, n_vars) )

		for f in range(0,n_offspring):

			cross_prop = np.random.uniform(0,1)
			offspring[f] = p1*cross_prop+p2*(1-cross_prop)

			# mutation
			for i in range(0,len(offspring[f])):
				if np.random.uniform(0 ,1)<=mutation:
					offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

			offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

			total_offspring = np.vstack((total_offspring, offspring[f]))

	return total_offspring




print('\nNEW EVOLUTION\n')

pop_p = np.random.uniform(dom_l, dom_u, (npop, n_vars))
pop_e = np.random.uniform(dom_l, dom_u, (npop, n_vars))

fit_pop_p = evaluate(pop_p, pop_e[0])

env.update_parameter('contacthurt','enemy')
fit_pop_e = evaluate(pop_e, pop_p[np.argmax(fit_pop_p)])

solutions = [pop_p, fit_pop_p, pop_e, fit_pop_e]
env.update_solutions(solutions)



def evolution(pop,fit_pop, best):

	offspring = crossover(pop,fit_pop)  # crossover
	fit_offspring = evaluate(offspring, best)   # evaluation
	pop = np.vstack((pop,offspring))
	fit_pop = np.append(fit_pop,fit_offspring)

	# selection
	fit_pop_cp = fit_pop
	fit_pop_norm = np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
	probs = (fit_pop_norm)/(fit_pop_norm).sum()
	chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
	chosen = np.append(chosen[1:], np.argmax(fit_pop))

	pop = pop[chosen]
	fit_pop = fit_pop[chosen]

	return pop, fit_pop


# evolution

switch = 4 # defined how many generations each agent has to evolve while the evolution of the other one is 'frozen'

for i in range(1, gens):

	# evolves one of the populations
	if env.contacthurt == "player":

		best = np.argmax(fit_pop_e)
		pop_p, fit_pop_p = evolution(pop_p, fit_pop_p, pop_e[best])

	else:

		best = np.argmax(fit_pop_p)
		pop_e, fit_pop_e = evolution(pop_e, fit_pop_e, pop_p[best])


	print( '\n GEN ',i, ' evolving ' ,env.contacthurt, ' - player mean ', np.mean(fit_pop_p) ,' - enemy mean ',  np.mean(fit_pop_e))


	# switches the evolution flag between player and enemy

	switch -= 1
	if switch == 0:

		switch = 4

		if env.contacthurt == "player":
			env.update_parameter('contacthurt','enemy')

		else:
			env.update_parameter('contacthurt','player')



	# saves simulation state
	solutions = [pop_p, fit_pop_p, pop_e, fit_pop_e]
	env.update_solutions(solutions)
	env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
