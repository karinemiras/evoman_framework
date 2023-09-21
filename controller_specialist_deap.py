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
from evolve.neural_net import NNController, NeuralNetwork

INPUT_SIZE = 20
HIDDEN = 10
OUTPUT_SIZE = 5
EXPERIMENT_NAME = 'nn_test'

if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

controller = NNController()
neural_net = NeuralNetwork(INPUT_SIZE, HIDDEN, OUTPUT_SIZE)
neural_net.load_weights(os.path.join(EXPERIMENT_NAME, 'weights.txt'))
env = Environment(experiment_name=EXPERIMENT_NAME,
                  speed="normal",
                  logs="off",
                  savelogs="no",
                  player_controller=controller,
                  visuals=True)

# tests saved demo solutions for each enemy
for en in range(2, 3):
    # Update the enemy
    env.update_parameter('enemies', [en])
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
    env.play(neural_net)
