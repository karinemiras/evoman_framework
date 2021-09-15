##################################################################################
# This file could be the main experiment script that we call from the command line
# e.g. "python run_experiment.py [arguments]"
#
# We can pass arguments such as population size, number of generations,
# enemy number, even something like which EA to use etc...
##################################################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from nn_controller import player_controller

# other imports
from evolution import initialize_generation, generate_next_generation, compute_fitness
import argparse
import IPython # nice for debugging : type 'IPython.embed()' on a line where you want to debug

# parses the arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--pop_size", type=int, required=False, default=100)
parser.add_argument("--num_gens", type=int, required=False, default=30)
parser.add_argument("--num-neurons", type=int, required=False, default=10)
parser.add_argument("--enemy", type=int, required=True)
args = parser.parse_args()

population_size = args.pop_size
num_generations = args.num_gens
num_hidden_neurons = args.num_neurons
enemy = args.enemy

# sets up experiment results folder for logs
experiment_name = "experiment_results/"+str(num_generations)+"x"+str(population_size)+"_enemy"+str(enemy)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initialize environment
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(num_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# total number of "genes" or weights in the neural network controller
num_genes = (env.get_num_sensors()+1)*num_hidden_neurons + (num_hidden_neurons+1)*5

# initialize first generation
current_generation = initialize_generation(population_size, num_genes)

# repeat for num_generations
for iteration in range(num_generations):

	# 1) run simulation on each individual in current_generation by calling compute_fitness(env, individual)

	# 2) call generate_next_generation using these results - this is where our genetic algorithm will come in

	# 3) keep track of important metrics such as fitness of every individual in every generation
	#    so that we can compute means, maximums etc. and make our plots

	# 4) also keep track of the best individual out of all generations

	pass


# high-level TODO tasks:
#		- add mechanism for saving progress, similar to what is inside of optimization_specialist_demo
#		- implement functions inside evolution.py
#		- maybe adding an additional outer loop that does the 10 runs>
#		- ...