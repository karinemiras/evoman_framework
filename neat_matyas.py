from argparse import ArgumentParser
import copy
import json
from pathlib import Path
import os
import sys

import numpy as np

# evoman
sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"
from controller import Controller
from environment import Environment

# neat
import neat
# from visualize import draw_net


class NeatController(Controller):
    def sigmoid_activation(self, x):
        return 1. / (1. + np.exp(-x))

    def control(self, inputs, net):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        output = np.array(net.activate(inputs))
        output = self.sigmoid_activation(output)
        return np.round(output)


# This is a singleton so that I don't have to use global variables
class Experiment:
    def initialize(name, enemy):
        Experiment.env = Environment(experiment_name=name,
                                     player_controller=NeatController(),
                                     enemies=[enemy],
                                     level=2,
                                     randomini='yes',
                                     savelogs='no',     # Save logs with NEAT instead
                                     clockprec='low',
                                     playermode="ai",
                                     speed="fastest",
                                     enemymode="static",
                                     contacthurt="player")
        Experiment.best_genome = None
        Experiment.best_gain = -101


def evaluate(genomes, config):
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        f, p, e, t = Experiment.env.play(pcont=net)
        if p - e > Experiment.best_gain:
            Experiment.best_genome = copy.deepcopy(genome)
            Experiment.best_gain = p - e
        genome.fitness = f


def parse_args(args):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--name', type=str, help='Experiment name', default='specialist_neat')
    parser.add_argument('--config', type=Path, help='Path to config file', default=Path('config_neat'))
    parser.add_argument('--checkpoint', type=Path, help='Path to checkpoint to load')
    parser.add_argument('--ch-interval', type=int, help='Checkpoint interval per generation', default=1)
    parser.add_argument('--max-gen', type=int, help='Maximum number of generations', default=30)
    parser.add_argument('--enemies', type=int, nargs='+', help='Enemies to use', default=[2, 6, 8])
    return parser.parse_args(args)


def main(args):
    parsed_args = parse_args(args)
    # Load NEAT configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, parsed_args.config)
    all_gains = {}

    # For each enemy
    for enemy in parsed_args.enemies:
        all_gains[enemy] = []

        # Run 10 independent experiments
        for i in range(10):

            log_path = Path(parsed_args.name, 'enemy-{}'.format(enemy), 'run-{}'.format(i))
            log_path.mkdir(parents=True, exist_ok=True)
            Experiment.initialize(str(log_path), enemy)

            pop = neat.Population(config)

            pop.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            pop.add_reporter(stats)
            pop.add_reporter(neat.Checkpointer(parsed_args.ch_interval,
                             filename_prefix=log_path.joinpath('checkpoint-')))

            # Run experiment
            winner = pop.run(evaluate, parsed_args.max_gen)
            # Save mean and max fitness by generation
            stats.save_genome_fitness(filename=str(log_path.joinpath('stats.csv')))
            # Save a visualization of winner network
            # draw_net(config, winner, directory=log_path, filename='winner')

            # Get gains for "best" solution
            gains = []
            for i in range(5):
                net = neat.nn.FeedForwardNetwork.create(Experiment.best_genome, config)
                f, p, e, t = Experiment.env.play(pcont=net)
                gains.append(p - e)
            all_gains[enemy].append(np.mean(gains))

    Path(parsed_args.name, 'gains').write_text(json.dumps(all_gains))


if __name__ == '__main__':
    main(sys.argv[1:])
