#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import os
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller
from evolve.neural_net import NNController

INPUT_SIZE = 20
HIDDEN = 10
OUTPUT_SIZE = 5
EXPERIMENT_NAME = 'nn_test'

if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# Update the number of neurons for this specific example
n_hidden_neurons = 0

# initializes environment for single objective mode (specialist)  with static enemy and ai player
# controller = player_controller(n_hidden_neurons)
controller = NNController(INPUT_SIZE, HIDDEN, OUTPUT_SIZE)
controller.load_weights(os.path.join(EXPERIMENT_NAME, 'weights.txt'))
env = Environment(experiment_name=EXPERIMENT_NAME,
                  playermode="ai",
                #   player_controller=controller,
                  speed="normal",
                  enemymode="static",
                  level=2,
                  visuals=True)

# tests saved demo solutions for each enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
    env.play(controller)
