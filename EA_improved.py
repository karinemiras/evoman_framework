"""
TO DO:
Change selecting best solution by gain. instead of fitness

Feng: statistical tests boxplots
Write report

# For final scores: use npop=100 and gens=100
"""

# imports framework
import sys
import os
import time
import csv

sys.path.insert(0, 'evoman')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import Environment
from demo_controller import player_controller

from argparse import ArgumentParser

from pathlib import Path

import numpy as np

N_HIDDEN_NEURONS = 10


# This is a singleton so that I don't have to use global variables
class Experiment:
    def initialize(name, enemy):
        Experiment.env = Environment(experiment_name=name,
                                     player_controller=player_controller(N_HIDDEN_NEURONS),  # Use demo_controller
                                     enemies=[enemy],
                                     level=2,
                                     randomini='yes',
                                     savelogs='no',  # Logging done manualy
                                     clockprec='low',
                                     playermode="ai",
                                     speed="fastest",
                                     enemymode="static",
                                     contacthurt="player")
        Experiment.best_genome = None
        Experiment.best_gain = -101


# Function to evaluate candidate solutions
def evaluate(individual):
    fitness, p, e, t = Experiment.env.play(pcont=individual)
    if p - e > Experiment.best_gain:
        Experiment.best_genome = individual
        Experiment.best_gain = p - e
    return fitness


def init_population(pop_size, n_hidden, n_input, n_output):
    genotype_len = (n_input + 1) * n_hidden + (n_hidden + 1) * n_output
    pop = np.random.uniform(lower_bound, upper_bound, size=(pop_size, genotype_len))
    return np.array(pop)


# Return the winner of a k-sized random tournament with replacement based on fitness.
# This function is used for parent selection.
def tournament_selection(pop, fitnesses, k):
    random_indices = np.random.randint(0, n_pop, k)
    winner_idx = np.argmax(fitnesses[random_indices])
    return pop[random_indices[winner_idx]]


# Adds gaussian noise to each allele with mutation_rate probability.
# To do: change step size to one often used in literature
def mutation(genotype, mutation_rate):
    for i in range(len(genotype)):
        if np.random.rand() < mutation_rate:
            genotype[i] += np.random.normal(0, 0.5)
    return genotype


def save_scores(scores, filepath):
    # open the file in the append mode
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(scores)
        f.close()


# def elitism(fitnesses, pop, k=1):
#     idx = (-fitnesses).argsort()[:k]
#     return pop[idx], fitnesses[idx]

# Round robin tournament for selection: q=5.
def round_robin_selection(parents, children, p_scores, c_scores, q=5):
    total_pop = np.concatenate((parents, children))
    scores = np.concatenate((p_scores, c_scores))

    wins = []
    # Compare each individual against Q others, count points
    for i in range(len(total_pop)):
        rand_ind = np.random.randint(0, len(total_pop), q)
        rand_scores = scores[rand_ind]
        points = sum(scores[i] >= rand_scores)
        wins.append(points)

    # Get top n_pop individuals based on number of wins
    top_q_ind = np.argpartition(wins, -n_pop)[-n_pop:]
    return total_pop[top_q_ind], scores[top_q_ind]


def uniform_crossover(parent_1, parent_2, cross_rate):
    if np.random.random() < cross_rate:
        alphas = np.random.random(len(parent_1))
        child = parent_1 * alphas + parent_2 * (1 - alphas)
        return child
    else:
        return parent_1 if np.random.random() < 0.5 else parent_2


def genetic_algorithm(n_generations, n_pop, cross_rate, mutation_rate, results_path, k):
    # Initialize population
    pop = init_population(n_pop, n_hidden=N_HIDDEN_NEURONS, n_input=20, n_output=5)

    # Keep best solution
    best_solution, best_fitness = 0, evaluate(pop[0])
    fitnesses = 0
    # Enumerate generations
    for gen in range(n_generations):

        # Evaluate population
        if gen == 0:
            fitnesses = np.array([evaluate(individual) for individual in pop])

        # Save plot results
        gen_mean = np.mean(fitnesses)
        gen_max = np.max(fitnesses)
        save_scores((gen_mean, gen_max), results_path)

        # Find new optimal solution
        for i in range(n_pop):
            if fitnesses[i] > best_fitness:
                best_solution, best_fitness = pop[i], fitnesses[i]
                print("Generation {}, new best fitness = {:.3f}".format(gen + 1, fitnesses[i]))

        # We do not need to generate children for last generation
        if not gen == n_generations - 1:
            children = []
            # Evolutionary Cycle (generate offspring)
            for i in range(n_pop):
                # Crossover and mutation (tournament, uniform alpha, gaussian noise)
                parent_1 = tournament_selection(pop, fitnesses, k)
                parent_2 = tournament_selection(pop, fitnesses, k)
                child = uniform_crossover(parent_1, parent_2, cross_rate)
                child = mutation(child, mutation_rate)
                children.append(child)
            children = np.array(children)

            # # Store elite
            # elite, elite_fitness = elitism(fitnesses, pop, k=1)

            # Evaluate offspring
            child_fitnesses = np.array([evaluate(x) for x in children])

            # Selection
            pop, fitnesses = round_robin_selection(pop, children, fitnesses, child_fitnesses)
            # pop[0], fitnesses[0] = elite, elite_fitness

    return [best_solution, best_fitness]


# define the total iterations
n_generations = 100
# define the population size
n_pop = 100
# crossover rate(typically in range (0.6, 0.9))
crossover_r = 0.9
# mutation rate(typically in range(1/chromosome_length, 1/pop_size))
# mutation_r = 1 / n_pop

# Arguably, increasing mutation rate should increase diversity.
# We allow model to run for more generations, so higher mutation rate
# only slows down convergence (hopefully no other side-effects?).
mutation_r = 0.2

# population initialization bounds
upper_bound = 1
lower_bound = -1

# tournament value
k = 2

if __name__ == '__main__':
    all_gains = {}

    # For each enemy
    enemies = [2]  # We do enemies 2,6,8
    # n_experiments = 10
    for enemy in enemies:

        # Run N independent experiments
        for i in [0, 1, 2, 3, 4]:

            log_path = Path('EA2', 'enemy-{}'.format(enemy), 'run-{}'.format(i))
            log_path.mkdir(parents=True, exist_ok=True)
            Experiment.initialize(str(log_path), enemy)
            results_path = os.path.join(log_path, 'improved-results-' + str(n_generations) + '.csv')

            # Remove previous experiment results
            if os.path.exists(results_path):
                os.remove(results_path)

            # Find and save best individual
            best_solution, best_fitness = genetic_algorithm(n_generations, n_pop, crossover_r, mutation_r, results_path,
                                                            k)
            print('Enemy {} - Run {} finished.'.format(enemy, i + 1))
            print('Best fitness = ', best_fitness)
            print('Best gain: ', Experiment.best_gain)
            solution_path = os.path.join(log_path, 'improved-solution-' + str(n_generations) + '.npy')
            # Save best individual based on gain
            np.save(solution_path, Experiment.best_genome)