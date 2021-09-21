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

tuning_parameter = "max_stagnation"
run_nr = 3  # number of runs
parameter_options = [2,3,4] # needs to be equal to the number of runs
experiment_name = 'test_run_NEAT'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)



def change_config_file(config, value):
    file = open(config, "r")
    list_of_lines = file.readlines()
    line = [idx for idx, s in enumerate(list_of_lines) if tuning_parameter in s][0]
    string = list_of_lines[line]
    name = string[:string.index("=")]
    list_of_lines[line] = name+"= "+str(value)+"\n"

    file = open(config, "w")
    file.writelines(list_of_lines)
    file.close()

    return config

def runs(config):
    for run in range(run_nr):
        config = change_config_file(config, parameter_options[run])
        apply_NEAT.run(config, run, experiment_name)







if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    runs(config_path)