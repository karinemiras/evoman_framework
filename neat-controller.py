import neat
from evoman.controller import Controller
from evoman.environment import Environment
import numpy as np
import os, time

experiment_name = 'neat-controller'
visuals = True
enemy = 5

def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))

class player_controller(Controller):
    def __init__(self, net):
        self.net = net  
    
    def control(self, inputs, _):
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        print(inputs)
        output = self.net.activate(inputs)
        return [1 if o > 0.5 else 0 for o in output]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player,
            enemymode="static",
            level=2,
            speed="fastest",
            fullscreen=False,
            visuals=visuals,
        ) 

        genome.fitness = env.play()[0]

def genome_to_vector(genome):
    weights = []
    biases = []
    
    for key in sorted(genome.connections.keys()):
        conn = genome.connections[key]
        if conn.enabled:
            weights.append(conn.weight)
            
    for node_key in sorted(genome.nodes.keys()):
        biases.append(genome.nodes[node_key].bias)
    
    return weights + biases

def save_genome_to_file(genome, filename):
    vector = genome_to_vector(genome)
    with open(filename, 'w') as file:
        for val in vector:
            file.write(str(val) + '\n')




def main():
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'sandor-neat_config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    generations = 2

    ini = time.time()
    
    winner = p.run(eval_genomes, generations)
    print(winner)


    fim = time.time() 
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    
    os.chdir(experiment_name)
    stats.save()

    filename = 'winner_genome'+str(enemy)+'.txt'
    save_genome_to_file(winner, filename)

    print('\n The vector of weights and biases was created for enemy '+str(enemy)+'.\n')

    
if __name__ == '__main__':
    main()