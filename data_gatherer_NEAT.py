import csv
import os
import sys

import neat
import numpy as np

from map_enemy_id_to_name import id_to_name

sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller


class PlayerController(Controller):
    def control(self, sensors, controller):
        return controller.activate(sensors)


def eval_genome_in_env(env: Environment):
    def eval_genome(genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return env.run_single(env.enemies[0], net, None)

    return eval_genome

def eval_genomes_in_env(env: Environment):
    eval_genome = eval_genome_in_env(env)

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            fitness, plife, elife, time = eval_genome(genome, config)
            genome.fitness = fitness

    return eval_genomes


class EvalEnvCallback(neat.reporting.BaseReporter):
    def __init__(
            self,
            eval_env_params,
            al_path,
            ar_path,
            raw_data_dir=None,
            lengths_prepend: list = [],
            rewards_prepend: list = [],
            n_eval_episodes: int = 5,
            eval_freq: int = 1,  # generations
    ):
        super(EvalEnvCallback, self).__init__()
        if not os.path.exists(raw_data_dir) and raw_data_dir is not None:
            os.makedirs(raw_data_dir)
        self.eval_env_params = eval_env_params
        self.al_path = al_path
        self.ar_path = ar_path
        self.lengths_prepend = lengths_prepend
        self.rewards_prepend = rewards_prepend
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.lengths = []
        self.rewards = []
        self.raw_data_dir = raw_data_dir
        self.generations = 0

    def post_evaluate(self, config, population, species, best_genome):
        eval_env = Environment(**self.eval_env_params, player_controller=PlayerController())
        self.generations = self.generations + 1
        if self.generations % self.eval_freq == 0:
            with open(f'{self.raw_data_dir}/wins.csv', mode='a') as wins_file:
                wins_writer = csv.writer(wins_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                with open(f'{self.raw_data_dir}/rewards.csv', mode='a') as rewards_file:
                    rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'',
                                                quoting=csv.QUOTE_NONNUMERIC)
                    eval_genome = eval_genome_in_env(eval_env)
                    wins = []
                    rs = []
                    ls = []
                    for j in range(self.n_eval_episodes):
                        fitness, plife, elife, time = eval_genome(best_genome, config)

                        rs.append(fitness)
                        ls.append(time)
                        if elife <= 0:
                            wins.append(1)
                        else:
                            wins.append(0)
                    self.lengths.append(np.mean(ls))
                    self.rewards.append(np.mean(rs))
                    wins_writer.writerow([self.generations, self.n_eval_episodes, ''] + wins)
                    rewards_writer.writerow([self.generations, self.n_eval_episodes, ''] + rs)

    def found_solution(self, config, generation, best):
            with open(self.al_path, mode='a') as eval_lengths_file:
                eval_lengths_writer = csv.writer(eval_lengths_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                with open(self.ar_path, mode='a') as eval_rewards_file:
                    eval_rewards_writer = csv.writer(eval_rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                    eval_lengths_writer.writerow(self.lengths_prepend+self.lengths)
                    eval_rewards_writer.writerow(self.rewards_prepend+self.rewards)


def run(config_file, environments, runs_start=0, runs=5, generations=100):
    for run in range(runs_start, runs_start+runs):
        print(f'Starting run {run}!')
        if sys.argv[1] == 'no':
            baseDir = f'FinalData/StaticIni/NEAT/run{run}'
        elif sys.argv[1] == 'yes':
            baseDir = f'FinalData/RandomIni/NEAT/run{run}'
        else:
            raise EnvironmentError()

        if not os.path.exists(baseDir):
            os.makedirs(baseDir)

        for enemy_id, enemy_envs in environments:
            enemyDir = f'{baseDir}/{id_to_name(enemy_id)}'
            if not os.path.exists(enemyDir):
                os.makedirs(enemyDir)

            for env_params, eval_env_params in enemy_envs:
                env = Environment(**env_params, player_controller=PlayerController())
                lengths_path = f'{enemyDir}/Evaluation_lengths.csv'
                rewards_path = f'{enemyDir}/Evaluation_rewards.csv'
                modelDir = f'{enemyDir}/models/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                rawDataDir = f'{enemyDir}/raw-data/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                if not os.path.exists(modelDir):
                    os.makedirs(modelDir)
                if not os.path.exists(rawDataDir):
                    os.makedirs(rawDataDir)
                l_prepend = [f'{id_to_name(enemy_id)}', ""]
                r_prepend = [f'{id_to_name(enemy_id)} ({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})', ""]

                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_file)

                p = neat.Population(config)
                p.add_reporter(neat.StdOutReporter(True))
                stats = neat.StatisticsReporter()
                p.add_reporter(stats)
                p.add_reporter(neat.Checkpointer(0, filename_prefix=f'{modelDir}/neat-checkpoint-'))
                p.add_reporter(EvalEnvCallback(
                    al_path=lengths_path,
                    ar_path=rewards_path,
                    lengths_prepend=l_prepend,
                    rewards_prepend=r_prepend,
                    raw_data_dir=rawDataDir,
                    n_eval_episodes=25,
                    eval_env_params=eval_env_params
                ))

                winner = p.run(eval_genomes_in_env(env), generations)

                print(
                    f'\nFinished {id_to_name(enemy_id)} ({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})')

if __name__ == '__main__':
    environments = [
        (
            n,
            [(
                dict(
                    enemies=[n],
                    weight_player_hitpoint=weight_player_hitpoint,
                    weight_enemy_hitpoint=1.0 - weight_player_hitpoint,
                    randomini=sys.argv[3],
                    logs='off',
                ),
                dict(
                    enemies=[n],
                    weight_player_hitpoint=1,
                    weight_enemy_hitpoint=1,
                    randomini=sys.argv[3],
                    logs='off',
                )
            ) for weight_player_hitpoint in [0.5]]
        )
        for n in [1, 2, 4, 7]
    ]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    run(config_path, environments, runs_start=int(sys.argv[1]), runs=int(sys.argv[2]), generations=100)
