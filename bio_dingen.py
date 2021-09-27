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
        fitness_smop = (1 + np.cos((2*np.pi/period) * g)) * (0.1*t) + (1 + np.cos((2*np.pi/period) * g + np.pi)) * (100-e+p / np.log(t))
        
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
    
#mutate a chromosome based on the mutation rate: the chance that a gene mutates
#and sigma: the average size of the mutation (taken from normal distribution)
def mutation(DNA, mutation_rate, sigma, m_b, m_m):
    length = len(DNA)
    
    #standard point mutations
    mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
    mutation_size = np.random.normal(0, 0.5*sigma**2+0.1, length)
    c1 = DNA + mutation_index*mutation_size
    
    #deletions (rare)
    if np.random.uniform(0, 1) < m_m*mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
        c1 = c1 * (mutation_index==False) + mutation_index * np.random.uniform(-1, 1, length)
    
    #insertions (rare)
    if np.random.uniform(0, 1) < m_m * mutation_rate:
        mutation_index = np.random.uniform(0, 1, length) < m_b+m_m*mutation_rate
        c1 = c1 * (mutation_index==False) + random.randint(2, 5) * c1 * mutation_index
    
    return c1

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
        child = random.choice(crossover(parents[p1[i]], parents[p2[i]]))
        
        #mutate based on parents fitness
        mutation_rate = 1-0.5*(fitness[p1[i]] + fitness[p2[i]])/(np.max(fitness)+1)
        sigma = 1-0.5*(fitness[p1[i]] + fitness[p2[i]])/(np.max(fitness)+1)
        child = mutation(child, mutation_rate, sigma, mutation_base, mutation_multiplier)
        
        #normalize between min-max
        minimum = -1
        maximum = 1
        for j in range(len(child)):
            if child[j]<minimum:
                child[j] = minimum
            elif child[j] > maximum:
                child[j] = maximum
        #child = (maximum-minimum)*(child-child.min())/(child.max()-child.min())+minimum
        children[i] = child
    return children