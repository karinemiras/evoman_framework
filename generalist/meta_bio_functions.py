# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:02:42 2021

@author: Sicco
Events file
"""

import numpy as np
import copy
import random


def utility_func(fitness_data ):
    utility = np.mean(fitness_data)

    return utility


def initialize_pop(input, pop_size):
    vars = np.zeros(( pop_size, len(input)))

    for idx, var in enumerate(input):
        if var == "mutation_baseline":
            vars[:,idx] = np.random.uniform(0, 1, pop_size)
        elif var == "mutation_multiplier":
            vars[:,idx] = np.random.uniform(0, 1, pop_size)
        elif var == "survival number":
            vars[:,idx] = np.random.uniform(0, 1, pop_size)

    return vars

def rescale_DNA(DNA):
    output = DNA

    output[:,0] = output[:,2]*0.3

    output[:,2] = output[:,2]*6
    output[:,2] = output[:,2].astype(int)

    return output

# uniform crossover (no position bias)
def crossover(p1, p2):
    length = len(p1)
    crossing_index = np.random.randint(2, size=length)
    c1 = p1 * crossing_index + p2 * (1 - crossing_index)
    c2 = p1 * (1 - crossing_index) + p2 * crossing_index

    return c1, c2


# mutate a chromosome based on the mutation rate: the chance that a gene mutates
# and sigma: the average size of the mutation (taken from normal distribution)
def mutation(DNA, mutation_rate, sigma, m_b, m_m):
    length = len(DNA)

    # standard point mutations
    mutation_index = np.random.uniform(0, 1, length) < m_b + m_m * mutation_rate
    mutation_size = np.random.normal(0, 0.5 * sigma ** 2 + 0.1, length)
    c1 = DNA + mutation_index * mutation_size

    # deletions (rare)
    if np.random.uniform(0, 1) < m_m * mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b + m_m * mutation_rate
        c1 = c1 * (mutation_index == False) + mutation_index * np.random.uniform(-1, 1, length)

    # insertions (rare)
    if np.random.uniform(0, 1) < m_m * mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b + m_m * mutation_rate
        c1 = c1 * (mutation_index == False) + random.randint(2, 5) * c1 * mutation_index

    return c1


def get_children(parents, surviving_players, utility, mutation_base, mutation_multiplier):
    children = copy.deepcopy(parents)

    # change all utility <0 to 0
    utility = np.array(utility)
    utility = utility + np.min(utility) +200
    utility = utility/np.max(utility)*100
    if not len(surviving_players) == len(utility):
        # pick parents based on utility (utility = weigth)
        parents_index = np.arange(0, len(parents), dtype=int)
        p1 = random.choices(parents_index, weights=utility, k=len(parents_index) - len(surviving_players))
        p2 = random.choices(parents_index, weights=utility, k=len(parents_index) - len(surviving_players))

        if len(surviving_players) > 0:
            p1 = np.hstack((surviving_players, p1))
            p2 = np.hstack((surviving_players, p2))

    else:
        p1 = surviving_players
        p2 = surviving_players

    # iterate to make children
    for i in range(len(parents)):
        # crossover the genes of parents and random choose a child
        child = random.choice(crossover(parents[p1[i]], parents[p2[i]]))

        # mutate based on parents utility
        mutation_rate = 1 - 0.5 * (utility[p1[i]] + utility[p2[i]]) / (np.max(utility) + 1)
        sigma = 1 - 0.5 * (utility[p1[i]] + utility[p2[i]]) / (np.max(utility) + 1)
        child = mutation(child, mutation_rate, sigma, mutation_base, mutation_multiplier)

        # normalize between min-max
        minimum = 0
        maximum = 1
        for j in range(len(child)):
            if child[j] < minimum:
                child[j] = minimum
            elif child[j] > maximum:
                child[j] = maximum
        # child = (maximum-minimum)*(child-child.min())/(child.max()-child.min())+minimum
        children[i] = child
    return children
