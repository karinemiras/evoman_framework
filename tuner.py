import numpy as np

from crossover import Crossover
from selection import Selection
from mutation import Mutation
from mutation_selection import MutationSelection


class Tuner:

    steps_count = 3
    evolutionary_algorithm_runs = 3

    def __init__(self, _evolutionary_algorithm):
        self.evolutionary_algorithm = _evolutionary_algorithm

    def run(self):
        for alpha in [0.6, 0.4, 0.3, 0.1]:

            print(f'Range: +/-{100*alpha}%')
            self.tune_parameter(Mutation.mutation_ratio, alpha, self.set_mutation_ratio)

            print('Mutation ratio - ', Mutation.mutation_ratio)
            self.tune_parameter(Selection.selection_ratio, alpha, self.set_selection_ratio)
            print('Selection ratio -', Selection.selection_ratio)
            self.tune_parameter(Crossover.offspring_ratio, alpha, self.set_offspring_ratio)
            print('Offspring ratio - ', Crossover.offspring_ratio)

            self.tune_parameter(MutationSelection.selection_ratio,
                                alpha, self.set_mutation_selection_ratio)
            print('Selection mutation ratio - ', MutationSelection.selection_ratio)

            print('\n\nCurrent results:')

            print('Mutation ratio - ', Mutation.mutation_ratio)
            print('Selection ratio -', Selection.selection_ratio)
            print('Offspring ratio - ', Crossover.offspring_ratio)

            print('Selection mutation ratio - ', MutationSelection.selection_ratio)
  
            print('\n\n\n')

    def tune_parameter(self, current_value, alpha, setter):
        bottom = (1 - alpha) * current_value
        top = (1 + alpha) * current_value
        best_value = 0
        best_score = 0


        for value in np.linspace(bottom, top, Tuner.steps_count):
            setter(value)
            score = self.test_algorithm()
            if(score > best_score):
                best_value = value
                best_score = score

        setter(best_value)

    def test_algorithm(self):
        score = 0
        for i in range(Tuner.evolutionary_algorithm_runs):
            _, best_fitness, _ = self.evolutionary_algorithm.run()
            score += best_fitness

        return score

    def set_mutation_ratio(self, value):
        Mutation.mutation_ratio = value

    def set_selection_ratio(self, value):
        Selection.selection_ratio = value

    def set_mutation_selection_ratio(self, value):
        MutationSelection.selection_ratio = value

    def set_offspring_ratio(self, value):
        Crossover.offspring_ratio = value
