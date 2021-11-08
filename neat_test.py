import os
import sys
import neat

sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller


class PlayerController(Controller):
    def control(self, sensors, controller):
        return controller.activate(sensors)


ENVIRONMENT = Environment(player_controller=PlayerController())


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        fitness, plife, elife, time = ENVIRONMENT.run_single(1, net, None)
        genome.fitness = fitness

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(0))

    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    #
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    print(stats)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat_test')
    run(config_path)