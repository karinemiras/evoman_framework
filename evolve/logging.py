import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


class DataVisualizer:
    def __init__(self, name):
        self.name = name
        self.generations = np.array([])
        self.best_fitness = np.array([])
        self.mean_fitness = np.array([])
        self.stats = []

    def gather(self, pop_fit, gen):
        self.generations = np.concatenate([self.generations, [gen]])
        self.best_fitness = np.concatenate([self.best_fitness, [np.max(pop_fit)]])
        self.mean_fitness = np.concatenate([self.mean_fitness, [np.mean(pop_fit)]])

        self.stats = np.stack([self.generations, self.best_fitness, self.mean_fitness])

        np.savetxt(f"{self.name}/stats.out", self.stats.T, delimiter=',', fmt='%1.2f')

    def draw_plot(self):
        mean_max, std_max, mean_mean, std_mean = self._read_data()
        # Set up
        sns.set()
        plt.figure()
        sns.set_theme(style="whitegrid")
        # This is where the actual plot gets made
        # Plot the responses for different events and regions
        sns.lineplot(x='Generation', y='Mean of Max', data=mean_max)
        ax = sns.lineplot(x='Generation', y='Mean of Mean', data=mean_mean)
        plt.yticks(np.arange(0, 100 + 1, 5.0))
        ax.fill_between(mean_max['Generation'], mean_max['Mean of Max'] - std_max['Std of Max'],
                        mean_max['Mean of Max'] + std_max['Std of Max'], alpha=0.2)
        ax.fill_between(mean_mean['Generation'], mean_mean['Mean of Mean'] - std_mean['Std of Mean'],
                        mean_mean['Mean of Mean'] + std_mean['Std of Mean'], alpha=0.2)

        ax.set_title(self.name)
        plt.legend(title='Legend', loc='lower right', labels=["Max", "StdMax", "Mean", "StdMean"])

        ax.set(xlabel='Generation', ylabel='Performance')

        # Save the plot in the plots folder
        plt.savefig(os.path.join(self.name, "plot.png"))

    def _read_data(self):
        file_path = os.path.join(self.name, "stats.out")
        print(file_path)
        data = pd.read_csv(file_path, names=["Generation", "Max", "Mean"])
        mean_max = data.groupby(['Generation'])['Max'].mean().reset_index(name='Mean of Max')
        std_max = data.groupby(['Generation'])['Max'].std().reset_index(name='Std of Max')
        mean_mean = data.groupby(['Generation'])['Mean'].mean().reset_index(name='Mean of Mean')
        std_mean = data.groupby(['Generation'])['Mean'].std().reset_index(name='Std of Mean')
        return mean_max, std_max, mean_mean, std_mean

