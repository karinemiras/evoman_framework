import sys
sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller
import numpy as np
import os


experiment_name = 'individual_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(0),  # no hidden layer
                  enemymode="static",
                  level=2,
                  speed="normal")

# 5 is a number of possible outputs
# env.get_num_sensors() is input size, +1 for bias
genome_length = 5 * (env.get_num_sensors() + 1)
genome = np.random.uniform(-1, 1, size=(genome_length,))
env.play(pcont=genome)
