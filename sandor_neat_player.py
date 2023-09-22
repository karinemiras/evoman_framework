import neat
from evoman.controller import Controller
from evoman.environment import Environment
import os
import pickle


experiment_name = 'neat-optimizer'
visuals = True
original_enemy = 5
run_number = 0
new_enemy = 5

env = Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemymode="static",
        level=2,
        speed="normal",
        sound='off',
        visuals=visuals
)

class player_controller(Controller):
    def __init__(self, net):
        self.net = net
    
    def control(self, inputs, _):
        print(inputs)
        output = self.net.activate(inputs)
        return [1 if o > 0.5 else 0 for o in output]

def run_winner(config, enemy, run_number, new_enemy): 
    env.enemies = [new_enemy]
    env.visuals = True
    env.speed = 'normal'

    with open(os.path.join(experiment_name, f'winner-{enemy}-{run_number}.pkl'), 'rb') as input_file:
        genome = pickle.load(input_file)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        player = player_controller(net)

        env.player_controller = player
        print(env.play()[0])


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'sandor_neat_config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_winner(config, original_enemy, run_number, new_enemy)

if __name__ == '__main__':
     main()