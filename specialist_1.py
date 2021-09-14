from typing import NewType
import numpy as np
from base_evolutionary_algorithm import EvolutionaryAlgorithm


def selection(fitness, population):
    i = np.flip(np.argsort(fitness), axis=None)

    return population[i[:(i.size // 5)], :]


def crossover(selected_individuals):
    selected_count = selected_individuals.shape[0]
    offsprings = np.array([])
    offspring_count = selected_count // 2
    genome_length = selected_individuals.shape[1]

    for i in range(offspring_count):
        partner = np.random.randint(selected_count)
        crossing_point = np.random.randint(genome_length)

        father_genes = selected_individuals[i, : crossing_point]
        mother_genes = selected_individuals[partner, crossing_point:]

        offspring = np.concatenate((father_genes, mother_genes), axis=None)
        offsprings = np.concatenate((offsprings, offspring), axis=None)

    mutants_count = selected_count // 10
    mutants = np.random.uniform(-1, 1, mutants_count * genome_length)

    newcomers_count = mutants_count + offspring_count
    return np.concatenate((offsprings, mutants), axis=None).reshape(newcomers_count, genome_length)


def insertion(fitness, population, newcomers):
    i = np.argsort(fitness, axis=None)
    population = np.delete(population, i[: newcomers.shape[0]], 0)
    return np.concatenate((population, newcomers))


evolutionaryAlgorithm = EvolutionaryAlgorithm(_experiment_name='solution1',
                                              _population_size=100,
                                              _generations_number=50,
                                              _selection=selection,
                                              _crossover=crossover,
                                              _mutation=1,
                                              _insertion=insertion)


controller = evolutionaryAlgorithm.findSolution()
