import neat
from evoman.controller import Controller
from evoman.environment import Environment
import os
import pickle
import csv


experiment_name = '7_Fixed_Structure'
visuals = True
original_enemy = 7
run_number = 9
new_enemy = 7
tipo = 'Fixed Structure'


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
    env.visuals = False
    env.speed = 'fastest'
    individual_gain = []
    for i in range(5):
        with open(os.path.join(experiment_name, f'winner-{enemy}-{run_number}.pkl'), 'rb') as input_file:
            genome = pickle.load(input_file)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            player = player_controller(net)

            env.player_controller = player
            print(env.play()[0])

        # Calculate the difference between player life and enemy life and print it
        player_life = env.get_playerlife()
        enemy_life = env.get_enemylife()
        life_difference = player_life - enemy_life
        individual_gain.append(life_difference)
        print(f'Player life {player_life} - Enemy life {enemy_life} is {life_difference}')
    print(individual_gain )
    with open(os.path.join(experiment_name, f'individual_gain-{original_enemy}-{run_number}-{new_enemy}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['1st Test', '2nd Test', '3rd Test', '4th Test','5th Test','Mean','Enemy Number','Run number','Type'])
        individual_gain.append(sum(individual_gain) / len(individual_gain))
        individual_gain.append(new_enemy)
        individual_gain.append(run_number)
        individual_gain.append(tipo)
        writer.writerow(individual_gain)
        


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'sandor_neat_config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_winner(config, original_enemy, run_number, new_enemy)


if __name__ == '__main__':
     main()