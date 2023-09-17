# Code from https://deap.readthedocs.io/en/master/examples/ga_onemax.html
# modified to run on our neural network and evoman objective. 
# go step-by-step through tutorial and see what differences you can
# spot between this code and code in the tutorial.

import random
import multiprocessing
import os

from deap import base
from deap import creator
from deap import tools
from evolve.neural_net import NNController
from evoman.environment import Environment


INPUT_SIZE = 20
HIDDEN = 10
OUTPUT_SIZE = 5
POP_SIZE = 100
NUM_ENEMIES = 8
NUM_GENS = 50
EXPERIMENT_NAME = 'nn_test'
# CXPB  is the probability with which two individuals are crossed
CXPB = 0.5
# MUTPB is the probability for mutating an individual
MUTPB = 0.2

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)
    
env = Environment(experiment_name=EXPERIMENT_NAME,
                  enemies=[NUM_ENEMIES],
                  playermode="ai",
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", NNController, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers
# define 'individual' to consist of of randomly initialized
# NeuralNet with params given by INPUT_SIZE, HIDDEN, OUTPUT_SIZE
toolbox.register("individual", creator.Individual, INPUT_SIZE, HIDDEN, OUTPUT_SIZE)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# setup game environment
# the goal ('fitness') function to be maximized
def evalFitness(individual):
    return env.play(pcont=individual)[0],

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalFitness)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=30)

#----------

def main():
    random.seed(2137)

    # create an initial population of POP_SIZE individuals 
    # (where each individual is a neural net)
    pop = toolbox.population(n=POP_SIZE)

    print("Start of evolution")

    # Evaluate and update fitness for the entire population
    update_fitness(toolbox.evaluate, pop)

    # # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    print_statistics(fits, len(pop))

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < NUM_GENS:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        update_fitness(toolbox.evaluate, invalid_ind)
        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        print_statistics(fits, len(pop))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    best_ind.save_weights(os.path.join(EXPERIMENT_NAME, 'weights.txt'))

def update_fitness(eval_func, pop):
    # Multiprocessing to make simulations run faster
    with multiprocessing.Pool() as pool:
        fitnesses = pool.map(eval_func, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return fitnesses

def print_statistics(fits, len_pop):
    print("  Evaluated %i individuals" % len_pop)
    mean = sum(fits) / len_pop
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / len_pop - mean**2)**0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

if __name__ == "__main__":
    main()
