################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

sys.path.insert(0, 'evoman')

# imports other libs
import numpy as np
import apply_NEAT 
import csv


experiment_name = 'test_run_NEAT'


run_nr = 1  # number of runs

def change_config_file(config):
    return config

def runs(config):
    for run in range(run_nr):
        config = change_config_file(config)
        apply_NEAT.run(config, run)






if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    runs(config_path)