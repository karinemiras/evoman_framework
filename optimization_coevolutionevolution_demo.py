############################################################################### 
# EvoMan FrameWork - V1.0 2016  			                      			  #
# DEMO : Neuroevolution - Genetic Algorithm with multilayer neural network.   #
# Author: Karine Miras        			                             		  #
# karine.smiras@gmail.com     				                     			  #
############################################################################### 

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller

# imports other libs 
import time
import numpy as np
from math import fabs,sqrt
import glob, os


# implements controller structure for player
class player_controller(Controller):



	def control(self, params,cont):

		params = (params-min(params))/float((max(params)-min(params))) # standardizes
		params = np.hstack( ([1.], params) )
		n_outs = [5] # number of output neurons (sprite actions)
		n_params = [len(params)] # number of input variables
		n_hlayers = 1	 # number of hidden layers
		n_lneurons = [10] # number of neurons in each hidden layer
		neurons_layers = []  # array of ann layers. Each position is a layer, and will contain another array in which each position will be a neuron.
		neurons_layers.append(params) # adds input neurons layer to the ann.


		weights = cont # reads weights of solution

		if n_hlayers==1:
			nh = n_params[0]*n_lneurons[0]
			weights1 = weights[:nh].reshape( (n_params[0], n_lneurons[0]) )
			output1 = 1./(1. + np.exp(-params.dot(weights1)))
			output1 = np.hstack( ([1],output1) )
			weights2 = weights[nh:].reshape( (n_lneurons[0]+1, n_outs[0]) )
			output = 1./(1. + np.exp(-output1.dot(weights2)))
		else:

			weights = weights.reshape( (n_params[0], n_outs[0]) )
			output = 1./(1. + np.exp(-params.dot(weights)))


		# takes decisions about sprite actions

		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0



		return [left, right, jump, shoot, release]



# implements controller structure for enemy
class enemy_controller(Controller):



	def control(self, params,cont):

		params = (params-min(params))/float((max(params)-min(params))) # standardizes
		params = np.hstack( ([1.], params) )
		n_outs = [5] # number of output neurons (sprite actions)
		n_params = [len(params)] # number of input variables
		n_hlayers = 1	 # number of hidden layers
		n_lneurons = [10] # number of neurons in each hidden layer
		neurons_layers = []  # array of ann layers. Each position is a layer, and will contain another array in which each position will be a neuron.
		neurons_layers.append(params) # adds input neurons layer to the ann.


		weights = cont # reads weights of solution

		if n_hlayers==1:
			nh = n_params[0]*n_lneurons[0]
			weights1 = weights[:nh].reshape( (n_params[0], n_lneurons[0]) )
			output1 = 1./(1. + np.exp(-params.dot(weights1)))
			output1 = np.hstack( ([1],output1) )
			weights2 = weights[nh:].reshape( (n_lneurons[0]+1, n_outs[0]) )
			output = 1./(1. + np.exp(-output1.dot(weights2)))
		else:

			weights = weights.reshape( (n_params[0], n_outs[0]) )
			output = 1./(1. + np.exp(-params.dot(weights)))


		# takes decisions about sprite actions

		if output[0] > 0.5:
			atack1 = 1
		else:
			atack1 = 0

		if output[1] > 0.5:
			atack2 = 1
		else:
			atack2 = 0

		if output[2] > 0.5:
			atack3 = 1
		else:
			atack3 = 0

		if output[3] > 0.5:
			atack4 = 1
		else:
			atack4 = 0



		return [atack1, atack2, atack3, atack4]




class environm(Environment):

	# implements fitness function
	def fitness_single(self):

		if self.contacthurt == "player":
			return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())

		else:
			return 0.9*(100 - self.get_playerlife()) + 0.1*self.get_enemylife() - np.log(self.get_time())


experiment_name = 'demo_co'

if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

# initializes simulation for coevolution evolution mode.
env = environm(experiment_name=experiment_name,
			   enemies=[2],
			   playermode="ai",
			   enemymode="ai",
			   player_controller=player_controller(),
			   enemy_controller=enemy_controller(),
			   level=2,
			   speed="fastest")


env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype/weights for perceptron phenotype network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test
#whan_vars = (env.get_num_sensors()+1)*5  # perceptron
n_vars = (env.get_num_sensors()+1)*10 + 11*5  # multilayer with 10 neurons
#n_vars = (env.get_num_sensors()+1)*50 + 51*5 # multilayer with 50 neurons
dom_u = 1
dom_l = -1
npop = 100
gens = 100
mutacao = 0.2
last_best = 0


# runs simulation
def simula(env,x1,x2):
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
def avalia(x1,x2):
	return np.array(map(lambda y: simula(env,y,x2), x1))


# tournment
def torneio(pop,fit_pop):
	c1 =  np.random.randint(0,pop.shape[0], 1)
	c2 =  np.random.randint(0,pop.shape[0], 1)

	if fit_pop[c1] > fit_pop[c2]:
		return pop[c1][0]
	else:
		return pop[c2][0]


# limits
def limites(x):

	if x>dom_u:
		return dom_u
	elif x<dom_l:
		return dom_l
	else:
		return x


	# crossover
def cruzamento(pop,fit_pop):

	total_filhos = np.zeros((0,n_vars))


	for p in range(0,pop.shape[0], 2):
		p1 = torneio(pop,fit_pop)
		p2 = torneio(pop,fit_pop)

		n_filhos =   np.random.randint(1,3+1, 1)[0]
		filhos =  np.zeros( (n_filhos, n_vars) )

		for f in range(0,n_filhos):

			cross_prop = np.random.uniform(0,1)
			filhos[f] = p1*cross_prop+p2*(1-cross_prop)

			# mutation
			for i in range(0,len(filhos[f])):
				if np.random.uniform(0 ,1)<=mutacao:
					filhos[f][i] =   filhos[f][i]+np.random.normal(0, 1)

			filhos[f] = np.array(map(lambda y: limites(y), filhos[f]))

			total_filhos = np.vstack((total_filhos, filhos[f]))

	return total_filhos




print('\nNEW EVOLUTION\n')

pop_p = np.random.uniform(dom_l, dom_u, (npop, n_vars))
pop_e = np.random.uniform(dom_l, dom_u, (npop, n_vars))

fit_pop_p = avalia(pop_p, pop_e[0])

env.update_parameter('contacthurt','enemy')
fit_pop_e = avalia(pop_e, pop_p[np.argmax(fit_pop_p)])

solutions = [pop_p, fit_pop_p, pop_e, fit_pop_e]
env.update_solutions(solutions)



def evolution(pop,fit_pop, best):

	filhos = cruzamento(pop,fit_pop)  # crossover
	fit_filhos = avalia(filhos, best)   # evaluation
	pop = np.vstack((pop,filhos))
	fit_pop = np.append(fit_pop,fit_filhos)

	# selection
	fit_pop_cp = fit_pop
	fit_pop_norm = np.array(map(lambda y: norm(y,fit_pop_cp), fit_pop)) # avoiding negative probabilities, as fitness is ranges from negative numbers
	probs = (fit_pop_norm)/(fit_pop_norm).sum()
	escolhidos = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
	escolhidos = np.append(escolhidos[1:], np.argmax(fit_pop))

	pop = pop[escolhidos]
	fit_pop = fit_pop[escolhidos]

	return pop, fit_pop


# evolution

switch = 4

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
	env.save_state(statesave)




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open('neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state