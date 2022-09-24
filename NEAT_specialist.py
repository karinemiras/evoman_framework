# imports framework
import sys, os
import neat
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
from NEAT_controller import NeatController

# np.set_printoptions(threshold=sys.maxsize) # remove console array truncation

experiment_name = "NEAT_specialist_enemy7"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes environment with ai player using random controller, playing against static enemy
# default environment fitness is assumed for experiment
game = Environment(experiment_name=experiment_name,
                   enemies=[7],
                   playermode="ai",
                   player_controller=NeatController(),
                   enemymode="static",
                   level=2,
                   speed="fastest")

# game.state_to_log()  # checks environment state

gens = 5 # generations

# all information from each game state
# all of this interacts with numpy.float64 out of Environment.py
def game_state(game, x):
    fitness, phealth, ehealth, time = game.play(pcont=x)
    return fitness

def genome_evaluation(genomes, config):
    for n_gen, x in genomes:
        x.fitness = 0
        x.fitness = game_state(game, x)

# engages the NEAT algorithm, see NEAT-Config.txt for details
def neat_execute(neat_config):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                neat_config)

    pop = neat.Population(config) # sets up the population

    pop.add_reporter(neat.StdOutReporter(True))
    s = neat.StatisticsReporter()
    pop.add_reporter(s)
    pop.add_reporter(neat.Checkpointer(gens))

    generations = pop.run(genome_evaluation, gens)
    print("\n Highest fitness:\n".format(generations))

if __name__ == '__main__':
    neat_dir = os.path.dirname(__file__)
    neat_path = os.path.join(neat_dir, "NEAT-config.txt")
    neat_execute(neat_path)