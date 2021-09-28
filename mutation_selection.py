
class MutationSelection:
    # This class is used to select individuals to be mutated
    """
        Creates the mutation set M

        Randomly choose individuals from the selection - BEFORE or AFTER CROSSOVER has been applied?
        
        Params:
        -------
        selection_ratio : relative number of individuals chosen from the selection, the selection after the crossover, or the whole population
            --> USE INDIVIDUALS FROM SELECTION or FROM AFTER CROSSOVER?


        Notes:
        ------
        parents.shape[0] : the dimensions of the parents are m * n, where m is the number of parents and n is the length of the genome
            - e.g. for 5 parents with genome_length 
    """

    selection_ratio = 0.3

    def only_parents(parents, offspring, population):
        selected_count = round(MutationSelection.selection_ratio * parents.shape[0])
        return parents[:selected_count, :]

    def only_offspring(parents, offspring, population):
        selected_count = round(MutationSelection.selection_ratio * offspring.shape[0])
        return offspring[:selected_count, :]

    def whole_population(parents, offspring, population):
        selected_count = round(MutationSelection.selection_ratio * population.shape[0])
        return population[:selected_count, :]
