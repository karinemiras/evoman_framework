import numpy as np


class Selection:
    """
        Creates the Selection set S

        Chooses individuals from the population for reproduction according to their fitness

        Params:
        ------
        selection_ratio : set in the specialist.py

        Methods:
        --------
        basic(fitness, population)
        tournament_selection(fitness, population)
    """

    selection_ratio = 0.3

    @staticmethod
    def basic(fitness, population):
        # Selects selection_ratio % of best individuals based on their fitness
        population_size = population.shape[0]
        selected_count = round(Selection.selection_ratio * population_size)
        sorted_indexes = np.flip(np.argsort(fitness), axis=None)
        selected_indexes = sorted_indexes[:selected_count]

        return population[selected_indexes, :]

    @staticmethod
    def tournament(fitness, population):
        population_size = population.shape[0]
        genome_length = population.shape[1]
        selected_count = round(Selection.selection_ratio * population_size)

        parents = np.array([])

        for i in range(selected_count):
            index1 = np.random.randint(population_size)
            index2 = np.random.randint(population_size)

            if fitness[index1] > fitness[index2]:
                parents = np.concatenate((parents, population[index1]))
            else:
                parents = np.concatenate((parents, population[index2]))

        return parents.reshape(selected_count, genome_length)
