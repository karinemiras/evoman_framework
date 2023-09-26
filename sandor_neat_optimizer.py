import neat
from evoman.controller import Controller
from evoman.environment import Environment
import os
import csv
import pickle
import psutil
import time


experiment = 'exp2_game7'
visuals = False
speed = 'fastest'
enemies = [7]
run_num = 1
runs = 10
generations = 30

# Declare global variables for monitoring
generation_no = []
memory_usage = []
cpu_usage = []
timep = []

if not os.path.exists(experiment):
    os.makedirs(experiment)

env = Environment(
        experiment_name=experiment,
        playermode="ai",
        enemymode="static",
        enemies=enemies,
        level=2,
        speed=speed,
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
def evaluate_genomes(genomes, config, generation):
    for genome_id, genome in genomes:
        start_time = time.time()
        genome.fitness = 4.0

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env.player_controller = player
        genome.fitness = env.play()[0]

        # Append data to the lists
        get_compdata_gen(start_time, run_num, genome_id, generation)


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

    def evaluate_genomes_with_generation(genomes, config):
        evaluate_genomes(genomes, config, p.generation)

    winner = p.run(evaluate_genomes_with_generation, generations)

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

    # Save CPU, memory, time, and generation data to a CSV file
    with open(os.path.join(working_directory, f'comp-data-{enemy}-{run_num}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['generation', 'time', 'memory_usage_MB', 'cpu_usage_percent'])

        for g, t, mem, cpu in zip(generation_no, timep, memory_usage, cpu_usage):
            writer.writerow([g, t, mem, cpu])

# Run function for the experiment
def run(config, enemies, runs):
    env.speed = speed
    env.visuals = visuals

    for enemy in enemies:
        env.enemies = [enemy]
        for run in range(0, runs):
            run_experiment(config = config, working_directory = experiment, enemy = enemy, run_num = run)

# Define computer usage data collector
def get_compdata_gen(start_time, run_num, genome_id, generation):
    end_time = time.time()
    # Collect CPU usage (percentage) and memory usage (in MB)
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Append data to lists
    generation_no.append(generation)
    cpu_usage.append(cpu_percent)
    memory_usage.append(memory_info.used / (1024 ** 2))  # Convert to MB
    timep.append(end_time - start_time)


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