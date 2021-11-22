import csv
import os
import random
import sys

# import neat
from deap import base, creator, tools, algorithms
import numpy as np

from map_enemy_id_to_name import id_to_name

sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller


# def eval_genome_in_env(env: Environment):
#     def eval_genome(genome, config):
#         net = neat.nn.RecurrentNetwork.create(genome, config)
#         return env.run_single(env.enemies[0], net, None)
#
#     return eval_genome
#
# def eval_genomes_in_env(env: Environment):
#     eval_genome = eval_genome_in_env(env)
#
#     def eval_genomes(genomes, config):
#         for genome_id, genome in genomes:
#             fitness, plife, elife, time = eval_genome(genome, config)
#             genome.fitness = fitness
#
#     return eval_genomes

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
    print("before: ", ind)
    for l in ind:
        for sl in l:
            sl = func(sl, **kwargs)

    print("after: ", ind)
    return ind,

def crossover_mlp(ind1, ind2, func, **kwargs):
    for i in range(len(ind1)):
        for j in range(len(ind1[i])):
            ind1[i][j], ind2[i][j] = func(ind1[i][j], ind2[i][j], **kwargs)

    return ind1, ind2

def run(environments, runs=5, generations=100):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_float_list', lambda n: [random.random() for _ in range(n)])
    toolbox.register('mate', crossover_mlp, func=tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', mut_mlp, func=tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=2)

    for run in range(runs):
        print(f'Starting run {run}!')
        baseDir = f'FullTime/DEAP/run{run}'

        if not os.path.exists(baseDir):
            os.makedirs(baseDir)

        for enemy_id, enemy_envs in environments:
            enemyDir = f'{baseDir}/{id_to_name(enemy_id)}'
            if not os.path.exists(enemyDir):
                os.makedirs(enemyDir)

            for env, eval_env in enemy_envs:
                modelDir = f'{enemyDir}/models/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                videoDir = f'{enemyDir}/videos/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                rawDataDir = f'{enemyDir}/raw-data/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
                if not os.path.exists(modelDir):
                    os.makedirs(modelDir)
                if not os.path.exists(videoDir):
                    os.makedirs(videoDir)
                if not os.path.exists(rawDataDir):
                    os.makedirs(rawDataDir)
                toolbox.register('individual', gen_multi_layer_network, creator.Individual, input=env.get_num_sensors(), output=5, layers=[10])
                toolbox.register('population', tools.initRepeat, list, toolbox.individual)
                toolbox.register('evaluate', evaluate, env=env)
                population = toolbox.population(100)
                algorithms.eaSimple(
                    population=population,
                    toolbox=toolbox,
                    cxpb=0.5,
                    mutpb=0.8,
                    ngen=generations,
                    verbose=True,
                )


    #
    #     for enemy_id, enemy_envs in enumerate(environments, start=1):
    #         enemyDir = f'{baseDir}/{id_to_name(enemy_id)}'
    #         if not os.path.exists(enemyDir):
    #             os.makedirs(enemyDir)
    #
    #         for env_params, eval_env_params in enemy_envs:
    #             env = Environment(**env_params)
    #             lengths_path = f'{enemyDir}/Evaluation_lengths.csv'
    #             rewards_path = f'{enemyDir}/Evaluation_rewards.csv'
    #             modelDir = f'{enemyDir}/models/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
    #             rawDataDir = f'{enemyDir}/raw-data/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
    #             if not os.path.exists(modelDir):
    #                 os.makedirs(modelDir)
    #             if not os.path.exists(rawDataDir):
    #                 os.makedirs(rawDataDir)
    #             l_prepend = [f'{id_to_name(enemy_id)}', ""]
    #             r_prepend = [f'{id_to_name(enemy_id)} ({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})', ""]
    #
    #             config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    #                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
    #                                  config_file)
    #
    #             p = neat.Population(config)
    #             p.add_reporter(neat.StdOutReporter(True))
    #             stats = neat.StatisticsReporter()
    #             p.add_reporter(stats)
    #             p.add_reporter(neat.Checkpointer(0, filename_prefix=f'{modelDir}/neat-checkpoint-'))
    #             p.add_reporter(EvalEnvCallback(
    #                 al_path=lengths_path,
    #                 ar_path=rewards_path,
    #                 lengths_prepend=l_prepend,
    #                 rewards_prepend=r_prepend,
    #                 raw_data_dir=rawDataDir,
    #                 n_eval_episodes=15,
    #                 eval_env_params=eval_env_params
    #             ))
    #
    #             winner = p.run(eval_genomes_in_env(env), generations)
    #
    #             print(
    #                 f'\nFinished {id_to_name(enemy_id)} ({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})')

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
                    show_display=True,
                ),
                Environment(
                    enemies=[n],
                    weight_player_hitpoint=1,
                    weight_enemy_hitpoint=1,
                    randomini='yes',
                    logs='off',
                    show_display=True,
                )
            ) for weight_player_hitpoint in [0.1, 0.4, 0.5, 0.6]]
        )
        for n in range(2, 5)
    ]

    run(environments, runs=1, generations=25)
