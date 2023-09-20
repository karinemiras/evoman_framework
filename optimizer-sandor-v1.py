import sys, os
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from deap import tools, creator, base, algorithms
from math import fabs,sqrt
import glob, os

# for which enemy
enemy = 8

# no visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'sandor-optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Environment
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

print("This is the simulation for enemy", enemy, "with", n_hidden_neurons, "hidden neurons.")

#env.state_to_log()

