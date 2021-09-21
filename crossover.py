import numpy as np


class Crossover:
    # This class contains diffrent crossover implementations

    # Keep in mind that offspring count CANNOT exceed popualtion count
    offspring_ratio = 1.5

    @staticmethod
    def basic(parents):
        parents_count = parents.shape[0]
        genome_length = parents.shape[1]

        offspring = np.array([])
        offspring_count = round(Crossover.offspring_ratio * parents_count)

        for i in range(offspring_count):
            mother = np.random.randint(parents_count)
            father = np.random.randint(parents_count)
            crossing_point = np.random.randint(genome_length)

            father_genes = parents[father, : crossing_point]
            mother_genes = parents[mother, crossing_point:]

            child = np.concatenate((father_genes, mother_genes), axis=None)
            offspring = np.concatenate((offspring, child), axis=None)

        return offspring.reshape(offspring_count, genome_length)

    def average(parents):
        parents_count = parents.shape[0]
        genome_length = parents.shape[1]

        offspring = np.array([])
        offspring_count = round(Crossover.offspring_ratio * parents_count)

        for i in range(offspring_count):
            mother = np.random.randint(parents_count)
            father = np.random.randint(parents_count)
            share = np.random.randint(1)

            child = share * parents[mother] + (1 - share) * parents[father]
            offspring = np.concatenate((offspring, child), axis=None)

        return offspring.reshape(offspring_count, genome_length)
