# Code from https://deap.readthedocs.io/en/master/examples/ga_onemax.html
# modified to run on our neural network and evoman objective.
# go step-by-step through tutorial and see what differences you can
# spot between this code and code in the tutorial.

import hydra
import os
import random
import multiprocessing
import numpy as np

from deap import base
from deap import creator
from deap import tools
from evolve.neural_net import NNController, NeuralNetwork
from evoman.environment import Environment
from evolve.logging import DataVisualizer

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

EXPERIMENT_NAME = "nn_test"
# ENEMY_IDX = [1, 2, 3, 4, 5, 6, 7, 8]
ENEMY_IDX = [1, 2, 3]

env = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=ENEMY_IDX,
    multiplemode="yes",
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=NNController(),
    visuals=False,
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    # create a data gatherer object
    logger = DataVisualizer(EXPERIMENT_NAME, config.evolve.selection_strategy)
    toolbox = prepare_toolbox(config)

    for i in range(config.train.num_runs):
        print(f"=====RUN {i + 1}/{config.train.num_runs}=====")
        new_seed = 2137 + i * 10
        best_ind = train_loop(toolbox, config, logger, new_seed, config.evolve.selection_strategy)

    #logger.draw_plots()
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    best_ind.save_weights(os.path.join(EXPERIMENT_NAME, "weights.txt"))


def prepare_toolbox(config):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", NeuralNetwork, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Structure initializers
    # define 'individual' to consist of randomly initialized
    # NeuralNet with params given by INPUT_SIZE, HIDDEN, OUTPUT_SIZE
    toolbox.register(
        "individual",
        creator.Individual,
        config.nn.input_size,
        config.nn.hidden_size,
        config.nn.output_size,
    )

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", eval_fitness)

    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinary, eta=config.evolve.eta_crossover)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=config.evolve.sigma_mutation,
                     indpb=config.evolve.indpb_mutation)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register(
        "parent_select", tools.selTournament, tournsize=config.evolve.selection_pressure
    )
    toolbox.register(
        "survivor_select",
        tools.selTournament,
        tournsize=config.evolve.selection_pressure,
    )
    # ----------
    return toolbox


# the goal ('fitness') function to be maximized
def eval_fitness(individual):
    return (env.play(pcont=individual)[0],)


def train_loop(toolbox, config, logger, seed, survivor_selection):
    random.seed(seed)
    np.random.seed(seed)
    # create an initial population of POP_SIZE individuals
    # (where each individual is a neural net)
    pop = toolbox.population(n=config.train.pop_size)

    # Variable keeping track of the number of generations
    g = 0

    print("Start of evolution")

    # Evaluate and update fitness for the entire population
    update_fitness(toolbox.evaluate, pop, config)

    # # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    print_statistics(fits, len(pop), len(pop))
    # save gen, max, mean, std
    #logger.gather_line(fits, g, survivor_selection)

    # Begin the evolution
    while g < config.train.num_gens:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.parent_select(pop, config.evolve.lambda_coeff * len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Shuffle the offspring
        random.shuffle(offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < config.evolve.cross_prob:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < config.evolve.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        update_fitness(toolbox.evaluate, invalid_ind, config)
        if config.evolve.selection_strategy == "comma":
            pop_len = len(pop)
            pop[:] = offspring
            pop = toolbox.survivor_select(pop, pop_len)
            pop = list(map(toolbox.clone, pop))
        elif config.evolve.selection_strategy == "plus":
            pop_len = len(pop)
            pop[:] = pop + offspring
            pop = toolbox.survivor_select(pop, pop_len)
            pop = list(map(toolbox.clone, pop))

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        print_statistics(fits, len(invalid_ind), len(pop))
        # save gen, max, mean
        #logger.gather_line(fits, g, survivor_selection)

    print("-- End of (successful) evolution --")
    return tools.selBest(pop, 1)[0]


def update_fitness(eval_func, pop, config):
    if config.train.multiprocessing == "true":
        cpu_count = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(processes=cpu_count) as pool:
            fitnesses = pool.map(eval_func, pop)
    else:
        fitnesses = map(eval_func, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return fitnesses


def print_statistics(fits, len_evaluated, len_pop):
    print("  Evaluated %i individuals" % len_evaluated)
    mean = sum(fits) / len_pop
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / len_pop - mean**2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)


if __name__ == "__main__":
    main()
