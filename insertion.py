import numpy as np


class Insertion:
    # This class contains diffrent insertion implementations

    @staticmethod
    def basic(fitness, population, offspring):
        # Replaces worse performing individuals with offspring
        offspring_count = min(offspring.shape[0], population.shape[0])
        sorted_indexes = np.argsort(fitness, axis=None)

        population = np.delete(population, sorted_indexes[:offspring_count], 0)
        return np.concatenate((population, offspring[:offspring_count]))
