import numpy as np


class Selection:
    # This class contains diffrent selection implementations

    # This value should be in (0, 1]
    selction_ratio = 0.3

    @staticmethod
    def basic(fitness, population):
        # Selects selection_ratio % of best individuals based on their fitness

        selected_count = round(Selection.selction_ratio * population.shape[0])
        sorted_indexes = np.flip(np.argsort(fitness), axis=None)
        selected_indexes = sorted_indexes[:selected_count]

        return population[selected_indexes, :]
