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
import meta_bio_functions as mbf

sys.path.insert(0, 'evoman')


class meta_evo_algorithm:
    '''
    The main Class containing the algorithm for a set of inputs.

    input               = [list] list of names of variables which are taken into account in the meta evo algorithm
    enemy               = [int] list of enemies (can be a single one) to train on
    experiment_nr       = [int] number of experiment runs, a experiment is a complete experiment of Y runs with X generations
    run_nr              = [int] total number of runs, a run is a complete iteration of X generations
    generations         = [int] number of generations per run
    population_size     = [int] size of the population
    fitter              = [string] name of the fitter to use
    experiment_name     = [string] the used name to save the data to
    n_hidden_neurons    = [int] number of hidden neurons in the hidden layer
    n_vars              = [int] total number of variables needed for the neural network
    total_data          = [list] place to save data to for later printing to csv
    n_sigmas            = [int] number of sigmas to use (only use 1 or 4)
    '''

    def __init__(self, input, n_hidden_neurons, run_nr_meta, enemy, run_nr, generations_meta, pop_size_meta, generations_runs, pop_size_runs,
                 fitter, repeats,  survival_number_meta, mutation_baseline_meta, mutation_multiplier_meta, run, cores='max', current_generation=0):

        self.input = input # which variables
        self.enemy = enemy  # which enemy
        self.run_nr_meta = run_nr_meta # number of experiments
        self.run_nr = run_nr  # number of runs
        self.generations_meta = generations_meta  # number of generations per meta
        self.pop_size_meta = pop_size_meta  # pop size meta
        self.generations_runs = generations_runs  # number of generations per run
        self.pop_size_runs = pop_size_runs  # pop size
        self.fitter = fitter
        self.repeats = repeats
        self.experiment_name = f'enemy_{enemy}_{fitter}'
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = len(input)
        self.total_data = []
        self.total_sigma_data = []
        self.best = []
        self.max_gain = -100 * len(enemy)
        self.n_sigmas = 4
        self.cores = cores
        self.survival_number = 4
        self.current_generation = current_generation
        self.survival_number_meta = survival_number_meta
        self.mutation_multiplier_meta = mutation_multiplier_meta
        self.mutation_baseline_meta = mutation_baseline_meta
        self.new = True
        self.run = run

        # make save folder
        if not os.path.exists(f'data_meta/{self.experiment_name}'):
            os.makedirs(f'data_meta/{self.experiment_name}')


    def execute_experiment(self):
        best_solutions = []
        # initialisation
        DNA = mbf.initialize_pop(self.input, self.pop_size_meta)

        for generation in range(self.generations_meta):
            utilities = []
            params = mbf.rescale_DNA(DNA.copy())
            for individual in range(self.pop_size_meta):
                mutation_baseline = params[individual, 0]
                mutation_multiplier = params[individual, 1]
                survival_number = params[individual, 2]

                print(DNA[individual,:])

                rmn.run_experiment(self.n_hidden_neurons, self.enemy, self.run_nr, self.generations_runs, self.pop_size_runs, mutation_baseline,
                                   mutation_multiplier, self.repeats, self.fitter, survival_number, self.cores, True)

                gains = []
                for run in range(self.run_nr):
                    df = pd.read_csv(f'data_normal/enemy_{self.enemy}_{self.fitter}/data_{run}.csv', skiprows = 1, header = None)
                    gains = np.append(gains, df.iloc[:,1])

                utility = mbf.utility_func(gains)
                utilities.append(utility)

            ##survival of X best players
            best_players = np.sort(utilities, axis=None)[len(utilities) - self.survival_number]
            indexes = np.where(utilities >= best_players)[0]

            surviving_players = indexes

            self.current_generation += 1

            best_sol = DNA[np.where(utilities == max(utilities))]

            best_solutions.append(best_sol)

            DNA = mbf.get_children(DNA, surviving_players, np.array(utilities), self.mutation_baseline_meta,
                               self.mutation_multiplier_meta)

        best_solutions = np.array(best_sol)
        np.savetxt(f"data_meta/{self.experiment_name}/best_solutions_meta_{run}.csv", best_solutions, delimiter=",")

def main():
    input = ["mutation_baseline", "mutation_multiplier", "survival number"]
    n_hidden_neurons = 10       #number of hidden neurons
    enemy = [5]               #which enemies
    run_nr_meta = 5           #number of meta experiments
    run_nr = 4                  #number of runs
    generations_meta = 10 #number of experiment generations
    pop_size_meta = 20 #meta pop size
    generations_runs = 20           #number of generations per run
    pop_size_runs = 5        #pop size
    survival_number_meta = 4
    mutation_baseline_meta = 0       #minimal chance for a mutation event
    mutation_multiplier_meta = 0.40  #fitness dependent multiplier of mutation chance
    repeats = 4
    fitter = 'standard'
    start = time.time()
    cores = 'max'
    new = True

    for run in range(run_nr_meta):
        meta_evo = meta_evo_algorithm(input, n_hidden_neurons, run_nr_meta, enemy, run_nr, generations_meta, pop_size_meta, generations_runs, pop_size_runs,
                 fitter, repeats, survival_number_meta, mutation_baseline_meta, mutation_multiplier_meta, run, cores)

        meta_evo.execute_experiment()
        #meta_evo.save_results(full=True)

if __name__ == "__main__":
    main()