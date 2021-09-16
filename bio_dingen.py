# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:02:42 2021

@author: pjotr
Events file
"""

import numpy as np
import copy
import random


#uniform crossover (no position bias)
def crossover(p1, p2):
    length = len(p1)
    crossing_index = np.random.randint(2, size=length)
    c1 = p1*crossing_index + p2*(1-crossing_index)
    c2 = p1*(1-crossing_index) + p2*crossing_index
    
    return c1, c2
    
#mutate a chromosome based on the mutation rate: the chance that a gene mutates
#and sigma: the average size of the mutation (taken from normal distribution)
def mutation(DNA, mutation_rate, sigma):
    length = len(DNA)
    mutation_index = np.random.uniform(0, 1, length) < 0.1+0.4*mutation_rate
    mutations = np.random.normal(0, sigma, length)
    c1 = DNA + mutation_index*mutations
    return c1

def get_children(parents, fitness):
    children = copy.deepcopy(parents)
    
    #change all fitness <0 to 0
    fitness = np.array(fitness)
    fitness = fitness*(fitness > 0)
    
    #normalize the fitness to make total_fitness = 1
    parents_index = np.arange(0, len(parents))
    p1 = random.choices(parents_index, weights=fitness, k=len(parents_index))
    p2 = random.choices(parents_index, weights=fitness, k=len(parents_index))
    
    #iterate to make children
    for i in range(len(parents)):
        #crossover the genes of parents and random choose a child
        child = random.choice(crossover(parents[p1[i]], parents[p2[i]]))
        
        #mutate based on parents fitness
        mutation_rate = 1-0.5*(fitness[p1[i]] + fitness[p2[i]])/(101)
        sigma = 1-0.5*(fitness[p1[i]] + fitness[p2[i]])/(101)
        child = mutation(child, mutation_rate, sigma)
        
        #normalize between min-max
        minimum = -1
        maximum = 1
        child = (maximum-minimum)*(child-child.min())/(child.max()-child.min())+minimum
        children[i] = child
        
    return children