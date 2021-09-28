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
import os
def prepare_NEAT_data(enemies):
    ### BOX PLOT DATA###
    df_gain = pd.DataFrame([])
    for idx, enemy in enumerate(enemies):
        folder = 'NEAT/final_results/enemy_'+str(enemy)+'/boxplot'  # folder with the data for boxplots
        folder =  'NEAT/final_results/enemy_1/boxplot'
        print(os.getcwd())
        os.listdir('NEAT/final_results/enemy_1/boxplot')
        df = pd.read_csv('NEAT/final_results/enemy_1/boxplot/data_0.csv')
        print(df)
        return

        gain_list = []
        for run in range(runs):
            df = pd.read_csv(f'{folder}/data_{run}.csv')
            gain = (df["p_health"] - df["e_health"]).mean()
            gain_list.append(gain)
        df_gain["enemy_"+str(enemy)] = gain_list

    print(df_gain)

def line_plots(runs, enemies):
    generations = 30


    for idx, enemy in enumerate(enemies):

        folder = 'final_results/enemy_' + str(enemy)  # folder with the data for boxplots

        max_values = np.array([])
        mean_values = np.array([])

        for run in range(runs):
            total_data = []


            df = pd.read_csv(f'{folder}/full_data_index_{run}.csv')

            total_data = np.array(df)

            #extract basic info and fitness data
            #enemy, generations, max_health = total_data[0]
            total_fitness_data = np.array(total_data[0:])
            enemy = int(enemy)
            generations = int(generations)
            population_size = len(total_fitness_data)

            #extract means and max
            max_values = np.append(max_values, total_fitness_data[:,0])
            mean_values = np.append(mean_values, total_fitness_data[:,1])



        #extract mean of max and mean of mean and their standard deviations
        max_values = max_values.reshape(runs, generations)
        mean_values = mean_values.reshape(runs, generations)
        mean_of_max = np.mean(max_values, axis=0)
        mean_of_mean = np.mean(mean_values, axis=0)
        std_of_max = np.std(max_values, axis=0)
        std_of_mean = np.std(mean_values, axis=0)



        #plotting
        x = range(generations)

        fig, ax = plt.subplots()
        ax.plot(x, mean_of_max, label='Max')
        ax.plot(x, mean_of_mean, label='Mean')
        ax.fill_between(x,
                        mean_of_max - std_of_max,
                        mean_of_max + std_of_max, alpha=0.2)
        ax.fill_between(x,
                        mean_of_mean - std_of_mean,
                        mean_of_mean + std_of_mean, alpha=0.2)
        ax.set_xlim(0, generations-1)
        ax.set_xticks(np.arange(0, generations+1, int(generations/10)))
        plt.ylim(0,100)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.title(f'Fitness over time against enemy {int(enemy)}', fontsize=14)
        plt.legend()
        plt.show()


def box_plots(runs, enemies):

    fig, ax = plt.subplots()
    for idx, enemy in enumerate(enemies):
        folder = 'final_results/enemy_'+str(enemy)+'/boxplot'  # folder with the data for boxplots
        f_list, gain_list = [], []

        for run in range(runs):
            df = pd.read_csv(f'{folder}/data_{run}.csv')
            f_list.append(df["fitness"].mean())
            gain = (df["p_health"] - df["e_health"]).mean()
            gain_list.append(gain)


        #create boxplot
        ax.boxplot(f_list, positions = [(idx+1)])

    ax.set_xticklabels(['enemy 1', 'enemy 5', 'enemy 8'], fontsize = 14)
    plt.title("Boxplots NEAT algorithm", fontsize =18, fontweight = 18)
    #plt.xlabel('Algorithm 1', fontsize=14)
    #plt.xticks([])
    plt.ylabel('Fitness', fontsize=14)
    plt.show()


os.chdir(r'C:\Users\Sicco\PycharmProjects\evoman_framework')
runs = 10 #amount of runs
enemies = [1, 5, 8]
# folder = 'final_results/enemy_1' #folder with info per generation and best weights
prepare_NEAT_data(enemies)
#line_plots(runs, enemies)

#box_plots(runs, enemies)
