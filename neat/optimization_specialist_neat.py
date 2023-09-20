# imports framework
import neat
import os
import pickle

from evoman.environment import Environment
from controller_neat import PlayerControllerNeat

# Change the enemy here, the winner will be saved in winner_neat_[ENEMY_IDX].pkl
# Keep in mind that if you run this file, the winner will be overwritten
ENEMY_IDX = 1
# Actual fitness values for each enemy (stored in winner_neat_[ENEMY_IDX].pkl)
# Update these values if you run this file
#ENEMY_1 fitness = 95.03
#ENEMY_2 fitness = 94.21
#ENEMY_3 fitness = 93.21
#ENEMY_4 fitness = 90.67
#ENEMY_5 fitness = 95.00
#ENEMY_6 fitness = 93.68
#ENEMY_7 fitness = 94.08
#ENEMY_8 fitness = 93.75

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# experiment_name = 'optimization_neat'
# if not os.path.exists(experiment_name):
#     os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name="optimization_specialist_neat",
                  enemies=[ENEMY_IDX],
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=PlayerControllerNeat(),  # you  can insert your own controller here
                  visuals=False)

# start writing your own code from here

# global variable to keep track of the generation
gen = 0


# runs simulation
def simulation(env, x):
    f, _, _, _ = env.play(pcont=x)
    return f


# evaluation
def eval_genomes(genomes, config):
    global gen
    gen += 1
    nets = []
    ge = []
    for _, genome in genomes:
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
    pickle.dump(winner, open('neat/winner_neat_' + str(ENEMY_IDX) + '.pkl', 'wb'))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward_neat.txt')
    run(config_path)
