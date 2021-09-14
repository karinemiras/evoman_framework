import sys
sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller
import numpy as np
import os


class EvolutionaryAlgorithm:
    def __init__(self,
                 _experiment_name,
                 _population_size,
                 _generations_number,
                 _selection,
                 _crossover,
                 _mutation,
                 _insertion):

        self.experiment_name = _experiment_name
        self.population_size = _population_size
        self.generations_number = _generations_number
        self.selection = _selection
        self.crossover = _crossover
        self.mutation = _mutation
        self.insertion = _insertion
        self.initialiseEnvironment()

    def findSolution(self):
        generation = 1
        self.initialisePopulation()
        while(generation <= self.generations_number):
            fitness = self.getFitness()
            selected_individuals = self.selection(fitness, self.population)
            newcomers = self.crossover(selected_individuals)
            self.population = self.insertion(
                fitness, self.population, newcomers)

            generation += 1

        return self.selection(fitness, self.population)[0]

    def getFitness(self):
        fitness = np.array([])

        for i in range(self.population_size):
            f, pl, el, t = self.env.play(pcont=self.population[i])
            fitness = np.append(fitness, f)

        return fitness

    def initialisePopulation(self):
        genome_length = 5 * (self.env.get_num_sensors() + 1)
        self.population = np.random.uniform(-1, 1,
                                            self.population_size * genome_length,)

        self.population = self.population.reshape(
            self.population_size, genome_length)

    def initialiseEnvironment(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.env = Environment(experiment_name=self.experiment_name,
                               enemies=[1],
                               playermode="ai",
                               player_controller=player_controller(0),
                               enemymode="static",
                               level=2,
                               speed="fastest")
