import numpy as np


class Crossover:
    # This class contains diffrent crossover implementations

    # Keep in mind that offspring count CANNOT exceed popualtion count

    """
    Implements different crossover variations

    IMPORTANT: offspring_count cannot exceed population count

    Methods
    -------
    basic : basic crossover method
    """

    

    @staticmethod
    def basic(parents):
        """
        Implements basic crossover method by using the offspring_ratio times the number of parents children
        
        Child is the sum of father genome (0:crossing_point) and mother genome (offspring_point:)
        Params
        ------
        parents : list of 2 parents for the offspring
        """
        offspring_ratio = 1.5
        parents_count = parents.shape[0]
        genome_length = parents.shape[1]

        offspring = np.array([])
        offspring_count = round(offspring_ratio * parents_count)

        for i in range(offspring_count):
            mother_index = np.random.randint(parents_count)
            father_index = np.random.randint(parents_count)
            crossing_point = np.random.randint(genome_length)

            father_genes = parents[father_index, : crossing_point]
            mother_genes = parents[mother_index, crossing_point:]

            child = np.concatenate((father_genes, mother_genes), axis=None)
            offspring = np.concatenate((offspring, child), axis=None)

        return offspring.reshape(offspring_count, genome_length)
