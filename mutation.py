import numpy as np


class Mutation:
    """
        Performs the mutation on the individuals in the Mating set


        Mutation ratio shows highest 
        Params:
        -------
        mutation_ratio : set by specialist.py


        Methods:
        --------
        basic(selected_group)           : draws perturbation value (int) from a discrete uniform distribution
        uniform_mutation(selected_group): draws perturbation value (float) from a uniform distribution
        success_rate(mutant, parent)    : TODO: proportion of successful mutations where child is superior to parent; helps evaluate mutation_rate parameter

    """

    @staticmethod
    def basic(selected_group):
        # selected_group is a subset of population selected to be mutated 
        # 
        # TODO: ITERATE THROUGH WHOLE POPULATION! PICK CANDIDATES WITH MUTATION_RATE PROBABILITY (draw value
        # from uniform distribution and check if value is smaller than probability - if yes then add to selected group --> CHANGE MUTATION_SELECTION)
        # Mutates mutation_ratio % of genes of selected individual, creating mutant
        # selected_group_count mutants will be created

        genome_length = selected_group.shape[1]
        mutated_genes_count = round(Mutation.mutation_ratio * genome_length)

        selected_group_count = selected_group.shape[0]
        mutants = np.array([])

        for i in range(selected_group_count):
            mutant = selected_group[np.random.randint(selected_group_count)]

            for j in range(mutated_genes_count):
                gene_index = np.random.randint(genome_length)
                mutant[gene_index] = np.random.randint(-1, 1)

            mutants = np.concatenate((mutants, mutant), axis=None)

        return mutants.reshape(selected_group_count, genome_length)


    def uniform_mutation(selected_group):

        genome_length = selected_group.shape[1]
        mutated_genes_count = round(Mutation.mutation_ratio * genome_length)

        selected_group_count = selected_group.shape[0]
        mutants = np.array([])

        for i in range(selected_group_count):
            mutant = selected_group[np.random.randint(selected_group_count)]

            for j in range(mutated_genes_count):
                gene_index = np.random.randint(genome_length)
                mutant[gene_index] = np.random.uniform(-1.0, 1.0)

            mutants = np.concatenate((mutants, mutant), axis=None)

        return mutants.reshape(selected_group_count, genome_length)