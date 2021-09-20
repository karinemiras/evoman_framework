import numpy as np


class Fitness:
    # This class contains diffrent selection implementations

    # max in around 21 (we use 2 norm, vectors are 105 long, values are between -1 and 1)
    neighbourhood_radius = 4

    @staticmethod
    def basic(population, env):
        fitness = np.array([])

        for individual in population:
            f, pl, el, t = env.play(pcont=individual)
            fitness = np.append(fitness, f)

        return fitness

    def niche(population, env):
        fitness = Fitness.basic(population, env)
        neighbours_count = np.array([])

        for individual in population:
            neighbours = 0
            for neighbour in population:
                if np.linalg.norm(individual - neighbour) < Fitness.neighbourhood_radius:
                    neighbours += 1
            neighbours_count = np.append(neighbours_count, np.sqrt(neighbours))

        print('Best in popularion:', np.amax(fitness))
        return fitness / neighbours_count
