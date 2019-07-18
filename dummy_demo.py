################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman') 
from environment import Environment

env = Environment() # initializes environment in default mode, with ai player and ai enemy using random controllers
env.play() # runs simulation 
env.state_to_log() # prints logs about simulation state


