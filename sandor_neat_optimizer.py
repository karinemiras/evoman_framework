import neat
from evoman.controller import Controller
from evoman.environment import Environment
import os
import csv
import pickle

experiment = 'exp1'
visuals = True
speed = 'fastest'
enemies = [1]
run_num = 1
runs = 10
generations = 20

if not os.path.exists(experiment):
    os.makedirs(experiment)

env = Environment(
        experiment_name=experiment,
        playermode="ai",
        enemymode="static",
        enemies=enemies,
        level=2,
        speed="normal",
        sound='off',
        visuals=visuals
)

class player_controller(Controller):
    # Usage of the default NEAT network
    def __init__(self, net):
        self.net = net
    
    # Control function
    def control(self, inputs, _):
        print(inputs)
        output = self.net.activate(inputs)
        #List comprehension to verify output
        return [1 if o > 0.5 else 0 for o in output]

#Evaluate each genome by the fitness function (default function from NEAT documentation)
def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env.player_controller = player
        genome.fitness = env.play()[0]


def run_experiment(config, working_directory, enemy, run_num = run_num):
    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    checkpointer = neat.Checkpointer(time_interval_seconds=60)
    checkpointer.filename_prefix = os.path.join(working_directory, f'neat-checkpoint-{enemy}-{run_num}-')
    p.add_reporter(checkpointer)
    
    winner = p.run(evaluate_genomes, generations)

    # Save stats to a csv with the run number
    with open(os.path.join(working_directory, f'stats-fitness-{enemy}-{run_num}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['generation', 'max_fitness', 'mean_fitness', 'stdev_fitness'])

        max_fitness = [c.fitness for c in stats.most_fit_genomes]
        mean_fitness = stats.get_fitness_mean()
        stdev_fitness = stats.get_fitness_stdev()

        for g in zip(range(0, len(max_fitness)), max_fitness, mean_fitness, stdev_fitness):
            writer.writerow([g[0], g[1], g[2], g[3]])

    print(winner)

    # Save the winner to a python pickle file
    with open(os.path.join(working_directory, f'winner-{enemy}-{run_num}.pkl'), 'wb') as output:
        pickle.dump(winner, output, 1)

# Run function for the experiment
def run(config, enemies, runs):
    env.speed = speed
    env.visuals = visuals

    for enemy in enemies:
        env.enemies = [enemy]
        for run in range(0, runs):
            run_experiment(config = config, working_directory = experiment, enemy = enemy, run_num = run)


def main():

    # Load config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'sandor_neat_config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Run the experiment
    run(config, enemies, runs=runs)


if __name__ == '__main__':
     main()