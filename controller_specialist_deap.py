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
ENEMY_IDX = 2


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

# tests saved demo solutions for ENEMY_IDX
print('\n LOADING SAVED SPECIALIST DEAP SOLUTION FOR ENEMY ' + str(ENEMY_IDX) + ' \n')
env.play(neural_net)