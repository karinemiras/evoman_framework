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

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

INPUT_SIZE = 20
HIDDEN = 10
OUTPUT_SIZE = 5
EXPERIMENT_NAME = "controller_generalist_deap"

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=EXPERIMENT_NAME,
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=player_controller(10),
                  visuals=False)

# tests solution for each enemy
print("\n LOADING SAVED SPECIALIST DEAP SOLUTION FOR ALL ENEMEIES \n")
sol = np.loadtxt(os.path.join("optimization_generalist_" + "island", "best_[2, 3, 5, 8].txt"))
num_of_defeated_enemies = 0
total_fitness = 0
total_gain = 0
for en in [2, 3, 5, 8]:
    # Update the enemy
    env.update_parameter("enemies", [en])

    f, p, e, t = env.play(sol)
    total_fitness += f
    total_gain += p - e
    if e == 0:
        print("Enemy " + str(en) + " defeated!\tGain: " + str(p - e))
        num_of_defeated_enemies += 1
    else:
        print("Enemy " + str(en) + " not defeated!\tGain: " + str(p - e))

print("\nTotal Firness: " + str(total_fitness) + "\nTotal Gain: " + str(total_gain))
print("\nAverage Fitness: " + str(total_fitness / 4) + "\nAverage Gain: " + str(total_gain / 8))
print("\n\nNumber of defeated enemies: " + str(num_of_defeated_enemies))
os.system('afplay /System/Library/Sounds/Sosumi.aiff')