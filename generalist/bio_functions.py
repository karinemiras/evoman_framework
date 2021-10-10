# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:02:42 2021

@author: pjotr
Events file
"""

import numpy as np
import copy
import random

def fitfunc(fitfunction, generations, g, t, e, p):

    if fitfunction == "standard":
        fitness_smop = 0.9*(100 - e) + 0.1*p - np.log(t)

    if fitfunction == "oscilation":
        period = .5*generations
        fitness_smop = (1 + np.cos((2*np.pi/period) * g)) * (0.1*t) + (1 + np.cos((2*np.pi/period) * g + np.pi)) * (100-e+p - np.log(t))

    if fitfunction == "exponential":
        fitness_smop =  100/(100-(0.9*(100 - e) + 0.1*p - np.log(t)))

    if fitfunction == "errfoscilation":
        if g < 0.5*generations:
            fitness_smop =  (0.01*t)**2
            if t == 1000:
                fitness_smop += .5*( 100 - e + p)
        else:
            fitness_smop = 150 - e + p - np.log(t)

    return fitness_smop


#uniform crossover (no position bias)
def crossover(p1, p2):
    length = len(p1)
    crossing_index = np.random.randint(2, size=length)
    c1 = p1*crossing_index + p2*(1-crossing_index)
    c2 = p1*(1-crossing_index) + p2*crossing_index

    return c1, c2


#weighted crossover based on fitness
def weighted_crossover(p1, p2, f1, f2):
    if f1+f2 == 0:
        w = 0.5
    else:
        w = f1/(f1+f2)
    c1 = w*p1 + (1-w)*p2

    return c1

#mutate a chromosome based on the mutation rate: the chance that a gene mutates
#and sigma: the average size of the mutation (taken from normal distribution)
def mutation(DNA, mutation_rate, sigma, m_b, m_m):

    #sigma function to map sigmas to the weights
    def sigma_func(length, sigma):

        #simple sigma function
        if len(sigma) == 1:
           sizes = np.random.normal(0, 1, length) * sigma[0]

        #multi sigma function
        #bias1 = [0:10]
        #weights1 = [10:210]
        #bias2 = [210:215]
        #weights2 = [215:]
        elif len(sigma) == 4:
            sizes           = np.zeros(length)
            sizes[:10]      = np.random.normal(0, 1, 10)* sigma[0]
            sizes[10:210]   = np.random.normal(0, 1, 200)* sigma[1]
            sizes[210:215]  = np.random.normal(0, 1, 5)* sigma[2]
            sizes[215:]     = np.random.normal(0, 1, 50)* sigma[3]

        return sizes

    length = len(DNA)

    #first mutate sigma(s)
    tau_ = 1/np.sqrt(2*length)
    tau = 1/np.sqrt(2*np.sqrt(length))
    if len(sigma) == 1:
        sigma = sigma * np.exp(np.sqrt(2)*tau_*np.random.normal(0, 1))
    elif len(sigma) == 4:
        sigma = sigma * np.exp(tau_*np.random.normal(0, 1) + tau*np.random.normal(0, 1, 4))

    #standard point mutations using new sigma(s)
    mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
    mutation_size = sigma_func(length, sigma)
    c1 = DNA + mutation_index*mutation_size

    #deletions (rare)
    if np.random.uniform(0, 1) < m_m*mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
        c1 = c1 * (mutation_index==False) + mutation_index * np.random.uniform(-1, 1, length)

    #insertions (rare)
    if np.random.uniform(0, 1) < m_m * mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
        c1 = c1 * (mutation_index==False) + random.randint(2, 5) * c1 * mutation_index

    return np.hstack((c1, sigma))

def get_children(parents, surviving_players, fitness, mutation_base, mutation_multiplier):
    children = copy.deepcopy(parents)

    #change all fitness <0 to 0
    fitness = np.array(fitness)
    fitness = fitness*(fitness > 0)

    if not len(surviving_players) == len(fitness):
        #pick parents based on fitness (fitness = weigth)
        parents_index = np.arange(0, len(parents), dtype=int)
        p1 = random.choices(parents_index, weights=fitness, k=len(parents_index)-len(surviving_players))
        p2 = random.choices(parents_index, weights=fitness, k=len(parents_index)-len(surviving_players))

        if len(surviving_players) > 0:
            p1 = np.hstack((surviving_players, p1))
            p2 = np.hstack((surviving_players, p2))

    else:
        p1 = surviving_players
        p2 = surviving_players

    #iterate to make children
    for i in range(len(parents)):
        #crossover the genes of parents and random choose a child
        #child = random.choice(crossover(parents[p1[i]], parents[p2[i]]))
        child = weighted_crossover(parents[p1[i]], parents[p2[i]], fitness[p1[i]], fitness[p2[i]])
        
        DNA   = child[:265]
        sigma = child[265:]

        #mutate based on parents fitness
        mutation_rate = 1-0.5*(fitness[p1[i]] + fitness[p2[i]])/(np.max(fitness)+1)
        child = mutation(DNA, mutation_rate, sigma, mutation_base, mutation_multiplier)

        #normalize between min-max
        minimum = -1
        maximum = 1
        min_sigma = -0.3
        max_sigma = 0.3
        thresh = 0.001
        for j in range(len(child)):
            if j < 265:
                if child[j]< minimum:
                    child[j] = minimum
                elif child[j] > maximum:
                    child[j] = maximum
            else:
                if abs(child[j]) < 0.001:
                    child[j] = 0.001
                if child[j]< min_sigma:
                    child[j] = min_sigma
                elif child[j] > max_sigma:
                    child[j] = max_sigma
                
        #child = (maximum-minimum)*(child-child.min())/(child.max()-child.min())+minimum
        children[i] = child
    return children
