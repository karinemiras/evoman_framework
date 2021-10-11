################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
import neat
import pandas as pd
import runscript_multi_normal as rmn
import time
sys.path.insert(0, 'evoman')


def main():
    n_hidden_neurons = 10       #number of hidden neurons
    enemies = [2, 6, 7, 8]               #which enemies
    run_nr = 1                  #number of runs
    generations = 300           #number of generations per run
    population_size = 100        #pop size
    mutation_baseline = 0       #minimal chance for a mutation event
    mutation_multiplier = 0.40  #fitness dependent multiplier of mutation chance
    repeats = 4
    fitter = 'standard'
    start = time.time()
    cores = 'max'
    new = True

    rmn.run_experiment(n_hidden_neurons, enemies, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, cores, new)


if __name__ == "__main__":
    main()