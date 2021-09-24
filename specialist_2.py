from tuner import Tuner
from base_evolutionary_algorithm import EvolutionaryAlgorithm

from fitness import Fitness
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


# HYPERPARAMS
population_size = 50
generations_number = 7

evolutionary_algorithm = EvolutionaryAlgorithm(_experiment_name='solution_2',
                                               _population_size=population_size,
                                               _generations_number=generations_number,
                                               _hidden_layer_size=10,
                                               _fitness=Fitness.basic,
                                               _selection=Selection.tournament_selection,
                                               _crossover=Crossover.basic,
                                               _mutation=Mutation.uniform_mutation,
                                               _mutation_selection=MutationSelection.only_parents,
                                               _insertion=Insertion.basic)

Mutation.mutation_ratio = 0.05
Crossover.offspring_ratio = 1
Selection.selection_ratio = 0.3
MutationSelection.selection_ratio = 0.5

tuner = Tuner(evolutionary_algorithm)
tuner.run()
