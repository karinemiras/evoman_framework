# imports framework
import sys, os
import pickle

import neat

from evoman.environment import Environment
from controller_neat import player_controller

# imports other libs
import numpy as np

# experiment_name = 'controller_specialist_neat'
# if not os.path.exists(experiment_name):
#     os.makedirs(experiment_name)


# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name="controller_specialist_neat",
                  speed="normal",
                  logs="off",
                  savelogs="no",
                  fullscreen=True,
                  player_controller=player_controller(),
                  visuals=True)

# tests saved neat solutions for selected enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    # Load the best genome saved in the winner_neat_[en].pkl file
    winner = pickle.load(open('winner_neat_' + str(en) + '.pkl', 'rb'))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward_neat.txt')
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
    env.play(pcont=net)