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
from demo_controller import player_controller

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

INPUT_SIZE = 20
HIDDEN = 10
OUTPUT_SIZE = 5
EXPERIMENT_NAME = 'nn_test'



if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

#controller = NNController()
#neural_net = NeuralNetwork(INPUT_SIZE, HIDDEN, OUTPUT_SIZE)
sol = np.loadtxt('nn_test/best.txt')

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=EXPERIMENT_NAME,
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=player_controller(10),
                  visuals=False)


# tests saved demo solutions for each enemy
#neural_net.load_weights(os.path.join(EXPERIMENT_NAME, 'weights_all.txt'))
print('\n LOADING SAVED SPECIALIST DEAP SOLUTION FOR ALL ENEMEIES \n')
total_fitness = 0
total_gain = 0
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    f, p, e, t = env.play(sol)
    total_fitness += f
    total_gain += p - e
    if e == 0:
        print("Enemy " + str(en) + " defeated!\tGain: " + str(p - e))

print('\nAverage Fitness: ' + str(total_fitness / 8) + '\nAverage Gain: ' + str(total_gain / 8) + '\n\n')
