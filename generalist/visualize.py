#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment_draw import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

# Update the number of neurons for this specific example
n_hidden_neurons = 10

experiment_name = 'visualize'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  randomini="yes")


# tests saved demo solutions for each enemy
for en in range(8):

	#Update the enemy
	env.update_parameter('enemies',[en+1])

	# Load specialist controller
	sol = np.loadtxt(f'data_normal/enemy_[1, 2, 5]_standard/best_sol_0.csv', delimiter=',')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
	env.play(sol[:265])
