################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import os
import sys

sys.path.insert(0, 'evoman')
from environment import Environment

experiment_name = 'test'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with human player and static enemies
for en in [4]:
    env = Environment(experiment_name=experiment_name,
                      enemymode='static',
                      speed="normal",
                      sound="off",
                      fullscreen=True,
                      use_joystick=True,
                      playermode='human',
                      show_display=True)
    env.update_parameter('enemies', [en])
    env.play()
