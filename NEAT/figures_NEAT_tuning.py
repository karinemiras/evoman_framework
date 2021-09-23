# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:32:38 2021

@author: pjotr
make figures
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = 'test_run_NEAT'
tuning_parameter = "Stagnation"
number_of_runs = 3
parameter_options = [2,3,4] # needs to be equal to the number of runs
enemy = 2

for run in range(number_of_runs):
    df = pd.read_csv(f'{folder}/fitness_data_{run}.csv')
    y = list(df["max"])
    x = list(df.index)
    plt.plot(x,y, label = "Stagnation param = "+str(parameter_options[run]))
plt.legend()
plt.xlabel('generation')
plt.ylabel('fitness')
plt.title(f'Fitness over time against enemy:{enemy}')
plt.show()


# total_fitness_data = []
# with open(f'{folder}/fitness_data_{run}.csv', newline='', encoding='utf-8') as f:
#     reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
#     for row in reader:
#         total_fitness_data.append(row)
#
# enemy, generations, max_health = total_fitness_data[0]
# total_fitness_data = total_fitness_data[1:]
#
# enemy = int(enemy)
# generations = int(generations)
# population_size = len(total_fitness_data)
#
# x = range(generations)
# total_fitness_data = np.array(total_fitness_data)
# fig, ax = plt.subplots()
# ax.plot(x, total_fitness_data[:,0])
# ax.plot(x, total_fitness_data[:,1])
# ax.fill_between(x,
#                 total_fitness_data[:,1] - total_fitness_data[:,2],
#                 total_fitness_data[:,1] + total_fitness_data[:,2], alpha=0.2)
# ax.set_xlim(0, generations-1)
# ax.set_xticks(range(generations))
# plt.ylim(0,100)
# plt.text(0.7*generations, 10, f'Max Fit  = {np.max(total_fitness_data[:,0]).round(2)}')
# plt.text(0.7*generations, 5, f'Max Life= {np.round(max_health, 2)}')
# plt.title(f'Fitness over time against enemy:{int(enemy)}')
# plt.show()