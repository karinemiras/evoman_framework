# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:32:38 2021

@author: pjotr
make figures
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

total_fitness_data = []
with open('testing_data.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        total_fitness_data.append(row)



x = range(len(total_fitness_data))
total_fitness_data = np.array(total_fitness_data)
fig, ax = plt.subplots()
ax.plot(x, total_fitness_data[:,0])
ax.plot(x, total_fitness_data[:,1])
ax.fill_between(x, 
                total_fitness_data[:,1] - total_fitness_data[:,2], 
                total_fitness_data[:,1] + total_fitness_data[:,2], alpha=0.2)