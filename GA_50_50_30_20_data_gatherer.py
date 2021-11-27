import csv
import os
import random
import sys
import pickle

# import neat
from deap import base, creator, tools, algorithms
import numpy as np

from map_enemy_id_to_name import id_to_name

sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))


def evaluate(individual, env: Environment):
    f, p, e, t = env.play(pcont=individual)
    return f,


class PlayerController(Controller):
    def control(self, sensors, controller):
        # Normalise input
        result = (sensors - min(sensors)) / float((max(sensors) - min(sensors)))
        for layer in controller:
            l = np.array(layer)
            result = sigmoid_activation(l.dot(result))

        return np.round(result).astype(int)


def gen_multi_layer_network(prototype, input, output, layers=[]):
    ls = [input, *layers, output]

    return prototype([
        np.array([
            [
                random.random() for _ in range(ls[i])
            ] for _ in range(ls[i+1])
        ], dtype=np.float) for i in range(len(ls)-1)
    ])


def mut_mlp(ind, func, **kwargs):
    for l in ind:
        for sl in l:
            func(sl, **kwargs)

    return ind,


def crossover_mlp(ind1, ind2, func, **kwargs):
    for i in range(len(ind1)):
        for j in range(len(ind1[i])):
            ind1[i][j], ind2[i][j] = func(ind1[i][j], ind2[i][j], **kwargs)

    return ind1, ind2


class EvalEnvCallback:
    def __init__(
            self,
            eval_env,
            model_dir=None,
            raw_data_dir=None,
            n_eval_episodes: int = 5,
            eval_freq: int = 1,  # generations
    ):
        if raw_data_dir is not None and not os.path.exists(raw_data_dir):
            os.makedirs(raw_data_dir)
        if model_dir is not None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.raw_data_dir = raw_data_dir
        self.model_dir = model_dir
        self.generations = 0

    def collect_data(self, population):
        with open(f'{self.model_dir}/generation-{self.generations}', mode='wb') as model_file:
            pickle.dump(population, model_file)
        [best_genome] = tools.selBest(population, 1)
        self.generations = self.generations + 1
        if self.generations % self.eval_freq == 0:
            wins = []
            rs = []
            ls = []
            for j in range(self.n_eval_episodes):
                fitness, plife, elife, time = self.eval_env.play(pcont=best_genome)

                rs.append(fitness)
                ls.append(time)
                if elife <= 0:
                    wins.append(1)
                else:
                    wins.append(0)
            with open(f'{self.raw_data_dir}/wins.csv', mode='a') as wins_file:
                wins_writer = csv.writer(wins_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                wins_writer.writerow([self.generations, self.n_eval_episodes, ''] + wins)
            with open(f'{self.raw_data_dir}/fitness.csv', mode='a') as rewards_file:
                rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'',
                                            quoting=csv.QUOTE_NONNUMERIC)
                rewards_writer.writerow([self.generations, self.n_eval_episodes, ''] + rs)

            return [np.mean(rs), np.mean(ls)]


def run(environments, runs=5, generations=100, population_size=100):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_float_list', lambda n: [random.random() for _ in range(n)])
    toolbox.register('mate', crossover_mlp, func=tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', mut_mlp, func=tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=2)

    stats = tools.Statistics(key=lambda ind: ind)

    for run in range(runs):
        print(f'Starting run {run}!')
        baseDir = f'FullTime/GA-50-50-30-20/run{run}'

        if not os.path.exists(baseDir):
            os.makedirs(baseDir)

        for enemy_id, enemy_envs in environments:
            enemyDir = f'{baseDir}/{id_to_name(enemy_id)}'
            if not os.path.exists(enemyDir):
                os.makedirs(enemyDir)

            for env, eval_env in enemy_envs:
                modelDir = f'{enemyDir}/models/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                # videoDir = f'{enemyDir}/videos/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                rawDataDir = f'{enemyDir}/raw-data/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                lengths_path = f'{enemyDir}/Evaluation_lengths.csv'
                rewards_path = f'{enemyDir}/Evaluation_rewards.csv'
                if not os.path.exists(modelDir):
                    os.makedirs(modelDir)
                # if not os.path.exists(videoDir):
                #     os.makedirs(videoDir)
                if not os.path.exists(rawDataDir):
                    os.makedirs(rawDataDir)

                evaluator = EvalEnvCallback(
                    eval_env=eval_env,
                    raw_data_dir=rawDataDir,
                    model_dir=modelDir,
                    n_eval_episodes=25,
                )

                stats.register('len_rew', evaluator.collect_data)

                toolbox.register('individual', gen_multi_layer_network, creator.Individual, input=env.get_num_sensors(), output=5, layers=[50, 50, 30, 20])
                toolbox.register('population', tools.initRepeat, list, toolbox.individual)
                toolbox.register('evaluate', evaluate, env=env)
                population = toolbox.population(population_size)
                population, logbook = algorithms.eaSimple(
                    population=population,
                    toolbox=toolbox,
                    cxpb=1,
                    mutpb=1,
                    ngen=generations,
                    stats=stats,
                    verbose=False,
                )

                averaged_data = np.rot90(np.array(logbook.select('len_rew')))

                with open(lengths_path, mode='a') as eval_lengths_file:
                    eval_lengths_writer = csv.writer(eval_lengths_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                    eval_lengths_writer.writerow(averaged_data[0])
                with open(rewards_path, mode='a') as eval_rewards_file:
                    eval_rewards_writer = csv.writer(eval_rewards_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                    eval_rewards_writer.writerow(averaged_data[1])

                print(
                    f'\nFinished {id_to_name(enemy_id)} ({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})')


if __name__ == '__main__':
    environments = [
        (
            n,
            [(
                Environment(
                    enemies=[n],
                    weight_player_hitpoint=weight_player_hitpoint,
                    weight_enemy_hitpoint=1.0 - weight_player_hitpoint,
                    randomini='yes',
                    logs='off',
                    player_controller=PlayerController(),
                    # show_display=True,
                ),
                Environment(
                    enemies=[n],
                    weight_player_hitpoint=1,
                    weight_enemy_hitpoint=1,
                    randomini='yes',
                    logs='off',
                    player_controller=PlayerController(),
                    # show_display=True,
                )
            ) for weight_player_hitpoint in [0.1, 0.4, 0.5, 0.6]]
        )
        for n in range(1, 9)
    ]

    run(environments, runs=5, generations=100, population_size=100)
