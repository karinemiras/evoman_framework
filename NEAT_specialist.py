# imports framework
import sys, os
import neat
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
from NEAT_controller import NeatController
import pickle

# np.set_printoptions(threshold=sys.maxsize) # remove console array truncation

# initializes environment with ai player using random controller, playing against static enemy
# default environment fitness is assumed for experiment

# game.state_to_log()  # checks environment state

class NEAT_Spealist():
    def __init__(self, env, gens, picklepath):
        neat_dir = os.path.dirname(__file__)
        neat_path = os.path.join(neat_dir, "NEAT-config.txt")
        
        self.game = env
        self.gens = gens
        self.picklepath = picklepath
        
        self.neat_execute(neat_path)
        
    # all information from each game state
    # all of this interacts with numpy.float64 out of Environment.py
    def game_state(self, game, x):
        fitness, phealth, ehealth, time = game.play(pcont=x)
        return fitness

    def genome_evaluation(self, genomes, config):
        for n_gen, x in genomes:
            x.fitness = 0
            x.fitness = self.game_state(self.game, x)

    # engages the NEAT algorithm, see NEAT-Config.txt for details
    def neat_execute(self, neat_config):
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    neat_config)

        # sets up the population
        pop = neat.Population(config)

        pop.add_reporter(neat.StdOutReporter(True))
        s = neat.StatisticsReporter()
        pop.add_reporter(s)
        pop.add_reporter(neat.Checkpointer(self.gens))

        # Run the algorithm
        winner = pop.run(self.genome_evaluation, self.gens)
        
        # save the winner
        with open(self.picklepath, "wb") as f:
            pickle.dump(winner,f)
            f.close()
        print("Saved the best solution at", self.picklepath)
        
        