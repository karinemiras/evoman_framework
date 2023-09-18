# imports framework
import pickle
import sys, os
import neat
import visualize_neat

from evoman.environment import Environment
from controller_neat import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# experiment_name = 'optimization_neat'
# if not os.path.exists(experiment_name):
#     os.makedirs(experiment_name)

# Change the enemy here, the winner will be saved in winner_neat_[en].pkl
# Keep in mind that if you run this file, the winner will be overwritten

#en = 1  # fitness = 95.03
#en = 2 # fitness = 94.21
#en = 3 # fitness = 93.21
#en = 4 # fitness = 90.67
#en = 5 # fitness = 95.00
#en = 6 # fitness = 93.68
#en = 7 # fitness = 94.08
#en = 8 # fitness = 93.75

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name="optimization_specialist_neat",
                  enemies=[en],
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=player_controller(),  # you  can insert your own controller here
                  visuals=False)

# start writing your own code from here

env.state_to_log()  # checks environment state

gen = 0


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def eval_genomes(genomes, config):
    global gen
    gen += 1
    nets = []
    ge = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        genome.fitness = simulation(env, net)
        # Uncomment to see the fitness of each genome in each generation
        # print("Gen: ", gen, "Fitness: ", genome.fitness)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 100 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))

    # Save the best genome in winner_neat.pkl
    pickle.dump(winner, open('winner_neat_' + str(en) + '.pkl', 'wb'))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward_neat.txt')
    run(config_path)

env.state_to_log()  # checks environment state
print()
