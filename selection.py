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

        selected_count = round(Selection.selection_ratio * population.shape[0])
        sorted_indexes = np.flip(np.argsort(fitness), axis=None)
        selected_indexes = sorted_indexes[:selected_count]

        return population[selected_indexes, :]

    def tournament_selection(fitness, population):
        pass
