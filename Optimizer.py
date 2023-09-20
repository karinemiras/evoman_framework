import sys
sys.path.insert(0, 'evoman')
import os
import random
import time
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller
# Set up environment without visuals
os.environ["SDL_VIDEODRIVER"] = "dummy" if True else os.environ["SDL_VIDEODRIVER"]

# Experiment setup
experiment_name = 'specialist experiment'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10
env = Environment(experiment_name=experiment_name, playermode="ai",
                  player_controller=player_controller(n_hidden_neurons), enemymode="static",
                  level=2, speed="fastest", randomini='yes')
env.state_to_log()

IND_SIZE = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Define fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_individual(individual, size):
    """Generate a new individual with random uniform values."""
    return individual(random.uniform(-1, 1) for _ in range(size))

def evaluate_individual(env, individual):
    """Evaluate the fitness of an individual."""
    f, _, _, _ = env.play(pcont=individual)
    return (f,)

toolbox.register("individual", generate_individual, creator.Individual, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual, env)
toolbox.register("mate", tools.cxUniform, indpb=0.05)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)

# Select parent selection method
selection_method = input("Select a type of parent selection between tournament and roulette:")

if selection_method == 'tournament':
    toolbox.register("select", tools.selTournament, tournsize=6)

elif selection_method == 'roulette':
    toolbox.register("select", tools.selRoulette)

# Setup statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Evolutionary algorithm parameters
params = {"MU": 100, "LAMBDA": 100, "cxpb": 0.6, "mutpb": 0.3, "ngen": 20, "nrep": 10}
enemies = ['enemy1', 'enemy2', 'enemy3']
best_of_gens = np.zeros((len(enemies), params["nrep"], IND_SIZE))

avrgs, stds, maxs, gens, enems, run = [], [], [], [], [], []

# Main optimization loop
for enemy_idx, enemy in enumerate(enemies):
    print(f'-------------- TRAINING AGAINST {enemy} --------------')
    env.update_parameter('enemies', [enemy_idx + 1])
    for rep in range(params["nrep"]):
        print(f'---------- TRAINING REPETITION # {rep + 1} ----------')
        hof = tools.HallOfFame(1)
        population = toolbox.population(n=params["MU"])
        _, logbook = algorithms.eaMuCommaLambda(population, toolbox, mu=params["MU"], lambda_=params["LAMBDA"],
                                                cxpb=params["cxpb"], mutpb=params["mutpb"], ngen=params["ngen"],
                                                halloffame=hof, stats=stats, verbose=False)
        best_of_gens[enemy_idx, rep, :] = hof[0]
        for gen in range(params["ngen"]):
            gens.append(gen + 1)
            enems.append(enemy_idx + 1)
            run.append(rep + 1)
            avrgs.append(logbook[gen]['avg'])
            maxs.append(logbook[gen]['max'])
            stds.append(logbook[gen]['std'])

# Save statistics and best solutions
data = pd.DataFrame({'average fitness': avrgs, 'max fitness': maxs, 'generations': gens,
                     'EA': [selection_method] * (params["ngen"] * params["nrep"] * len(enemies)), 'enemy': enems})
print(data)

os.makedirs('data', exist_ok=True)
data.to_csv(f'dataframe/Dataframe_EA{selection_method}')

os.makedirs('best_solutions', exist_ok=True)
for idx, enemy in enumerate(enemies):
    np.save(f'best_solutions/BestSolutions_enemy{idx + 1}_EA{selection_method}', best_of_gens[idx, :, :])
