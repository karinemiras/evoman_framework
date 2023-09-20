import os
import neat
import numpy as np
from evoman.environment import Environment
from evoman.controller import Controller
env=Environment()

class player_controller(Controller):
    def __init__(self, n_hidden_neurons):
        self.n_hidden_neurons=n_hidden_neurons

experiment_name = 'neat_experiment'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10

env = Environment(
    experiment_name=experiment_name,
    enemies=[1], 
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
)

config_path = 'sandor-neat_config.ini'

def run_game(env, neural_net):
    fitness, p, e, t = env.play(pcont=neural_net)
    return fitness

def fitness_default(enemylife, playerlife, time):
    return 0.9*(100 - enemylife) + 0.1*playerlife - np.log(time)

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def evaluate_genome(genomes, config):
    fitness_scores = []

    for genome in genomes:
        genome.fitness = simulation(env, genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    for _ in range(10):  
        fitness = run_game(env, net)  
        enemylife = env.get_enemylife()
        playerlife = env.get_playerlife() 
        time = env.get_time() 

        fitness_default_value = fitness_default(enemylife, playerlife, time)
        fitness_scores.append(fitness_default_value)

    mean_fitness = np.mean(fitness_scores)
    
    return mean_fitness


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(evaluate_genome, 30)
    with open('best_neural_net.txt', 'w') as f:
        f.write(str(winner))

if __name__ == '__main__':
    main()
