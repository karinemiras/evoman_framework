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

def make_figures(save_figures, show_figure, name_experiment, parameter_options, parameter_names, number_of_runs, enemy):
    list_means = []
    list_max = []
    list_sdev = []



    count_run = 0
    sum_of_maxima = 0

    for  parameter in parameter_options:
        for count_run in range(number_of_runs):
            df = pd.read_csv(f'{name_experiment}/fitness_data_{parameter}_run_{count_run}.csv')

            sum_of_maxima += df["max"].max()

        list_max.append(sum_of_maxima/number_of_runs)
        sum_of_maxima = 0

    parameter_options = [str(i) for i in parameter_options]
    range_list = np.arange(1,(len(list_max)+1),1)
    range_list = [str(i) for i in range_list]

    plt.bar(range_list, list_max )
    #plt.legend()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title(f'Fitness over time against enemy:{enemy}')

    if save_figures:
        plt.savefig("")
    if show_figure:
        plt.show()

