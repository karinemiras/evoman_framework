from base_evolutionary_algorithm import EvolutionaryAlgorithm
from crossover import Crossover
from selection import Selection
from insertion import Insertion
from mutation import Mutation
from mutation_selection import MutationSelection


# Before you run evolutionary algorithm you can adjust following variables:

# Crossover.offspring_ratio - says what's the offspring/parents ratio, default 1.5
# Selection.selection_ratio - says how many % of population should be selected, default 0.3
# Mutation.mutation_ratio - says how many % of genes will be mutated, default 0.1
# MutationSelection.selection_ratio - says how many % of given group should be selected, default 0.3


Mutation.mutation_ratio = 0.3
MutationSelection.selction_ratio = 0.5

# HYPERPARAMS
population_size = 10
generations_number = 5

evolutionary_algorithm = EvolutionaryAlgorithm(_experiment_name='solution_1',
                                               _population_size=population_size,
                                               _generations_number=generations_number,
                                               _selection=Selection.basic,
                                               _crossover=Crossover.basic,
                                               _mutation=Mutation.basic,
                                               _mutation_selection=MutationSelection.only_offspring,
                                               _insertion=Insertion.basic)


best, best_fitness = evolutionary_algorithm.run()
print(best)
