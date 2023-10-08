#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                                 		  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        general solution for enemies (games)                                         #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller
from evolve.neural_net import NNController

# imports other libs
import numpy as np

experiment_name = "controller_generalist_demo"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(
    experiment_name=experiment_name,
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    speed="normal",
    enemymode="static",
    level=2,
    visuals=True,
)

sol = np.loadtxt("nn_test/weights.txt")
print("\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n")

# tests saved demo solutions for each enemy
for en in range(1, 4):
    # Update the enemy
    env.update_parameter("enemies", [en])

    print(env.play(sol))

print("\n  \n")
