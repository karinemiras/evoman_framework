import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.ma.extras import average

DEBUG = True

class Experiment:

    # avg_fitness = np.array([])  # arrays of average fitnesses of every run of generation
    best_solutions = np.array([[]])  # 2D array which stores best member/solution of every run
    best_solutions_fitness = np.array([])  # array of best solution's fitness of every run

    def __init__(self, _evolutionary_algorithm):
        self.evolutionary_algorithm = _evolutionary_algorithm

    def run_experiment(self, experiments):
        # store average fitness per generation in array
        avg_fitness_gen = np.array([])

        for i in range(experiments):
            best, best_fitness, avg_generation_fitness = self.evolutionary_algorithm.run()
            avg_fitness_gen = np.append(avg_fitness_gen, np.average(avg_generation_fitness)) # assign average of average fitness to avg_fitness_gen
            self.best_solutions = np.append(self.best_solutions, best)
            self.best_solutions_fitness = np.append(self.best_solutions_fitness, best_fitness)

            if DEBUG: print(f'EXPERIMENT NUMBER {i+1}: Average generation fitness: {avg_fitness_gen, avg_fitness_gen.shape} \n\n\n Fitness of the best solutions {self.best_solutions_fitness}')
        # Plot the results of all experimental runs
        self.line_plot(avg_fitness_gen, experiments)

        # Save best of best_solutions to wonderful CSV file <3

    def line_plot(self, average_fitness_generation, num_experiments):
        """
            Implements the plotting of the generational average fitness value & the highest fitness value
            Is called after each experiment

        """
        # 1. ASSIGN ARRAY OF AVERAGE FITNESSES OVER ALL GENERATIONS OVER EACH EXPERIMENT TO DATAPOINTS
        data_points = average_fitness_generation

        # 2. FILL ARRAY WITH VALUES CORRESPONDING TO NUMBER OF EXPERIMENTS
        array_experiments = np.array([i+1 for i in range(num_experiments)]) # list comprehension
        
        if DEBUG: print(f'Experiments array = {array_experiments, array_experiments.shape}; Average fitnesses array = {data_points, data_points.shape}')

        # 3. PLOT DATAPOINTS AGAINST EXPERIMENT_NUMBER
        plt.plot(array_experiments, data_points, label="average generation fitness per experiment")

        # 4. PLOT PARAMS
        xint = range(min(array_experiments), math.ceil(max(array_experiments)) + 1)
        plt.xticks(xint)  # set y-axis to only integer values
        plt.xlabel('generation')
        plt.ylabel('average fitness')

        plt.show()

