################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
import neat
import pandas as pd
sys.path.insert(0, 'evoman')
from environment import Environment
from specialist_controller import NEAT_Controls



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'NEAT_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
enemy = 2                   #which enemy
generations = 2            #number of generations per run
total_fitness_data = []
children_index = []
children_data = []
max_health = 0
generation = 0
# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=NEAT_Controls(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state


# runs simulation



def eval_genomes(genomes, config):
    fitness_array = []
    fitness_array_smop = []
    global generation
    generation += 1
    for genome_id, genome in genomes:
        genome.fitness = 0

        f, p, e, t = env.play(pcont=genome)

        fitness_new = 0.9 * (100 - e) + 0.1 * p - np.log(t)
        fitness_smop = (100 / (100 - (0.9 * (100 - e) + 0.1 * p - np.log10(t))))

        children_index.append([generation, f, p, e, t])
        #children_data.append(genome)

        fitness_array.append(fitness_new)
        fitness_array_smop.append(fitness_smop)

        genome.fitness = fitness_new

    total_fitness_data.append([np.max(fitness_array),
                               np.mean(fitness_array),
                               np.std(fitness_array)])



def run(config_file, run):
    """
    runs the NEAT algorithm to train a neural network to play mega man.
    It uses the config file named config-feedforward.txt. After running it stores it results in CSV files.
    """

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to generations generations.
    winner = p.run(eval_genomes, generations)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # stats to csv
    print(total_fitness_data)
    total_fitness_data_df = pd.DataFrame(total_fitness_data, columns = [enemy, generations, max_health])
    print(total_fitness_data_df)
    total_fitness_data_df.to_csv(f'{experiment_name}/fitness_data_{run}.csv', index = False)

    children_index_df = pd.DataFrame(children_index, columns = ['generation', 'fitness', 'p_health',
                         'e_health', 'time'])
    children_index_df.to_csv(f'{experiment_name}/full_data_index_{run}.csv', index = False)

    # children_data_df = pd.DataFrame(children_data)
    # children_data_df.to_csv(f'{experiment_name}/full_data_{run}.csv', index = False)

    total_fitness_data.clear()
    children_index.clear()
    children_data.clear()
    global generation
    generation = 0


