import itertools
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


class DataVisualizer:
    def __init__(self, name, survivor_selection):
        self.name = name
        self.survivor_selection = survivor_selection
        self.generations = np.array([])
        self.best_fitness = np.array([])
        self.mean_fitness = np.array([])
        self.winner = np.array([])
        self.gain = np.array([])
        self.stats_line = []
        self.stats_box = []

    def gather_line(self, pop_fit, gen, survivor_selection):
        self.generations = np.concatenate([self.generations, [gen]])
        self.best_fitness = np.concatenate([self.best_fitness, [np.max(pop_fit)]])
        self.mean_fitness = np.concatenate([self.mean_fitness, [np.mean(pop_fit)]])

        self.stats_line = np.stack([self.generations, self.best_fitness, self.mean_fitness])

        np.savetxt(f"{self.name}/stats_line_" + survivor_selection + ".out", self.stats_line.T, delimiter=',',
                   fmt='%1.6f')

    def gather_box(self, winner, gain, survivor_selection):
        self.winner = np.concatenate([self.winner, [winner]])
        self.gain = np.concatenate([self.gain, [gain]])

        self.stats_box = np.stack([self.winner, self.gain])

        np.savetxt(f"{self.name}/stats_box_" + survivor_selection + ".out", self.stats_box.T, delimiter=',',
                   fmt='%1.6f')

    def draw_plots(self):
        # Read data from file
        survivor_selection = "comma"
        mean_max, std_max, mean_mean, std_mean = self._read_data_line(survivor_selection)
        # Set up line plot
        sns.set()
        palette = itertools.cycle(sns.color_palette())
        plt.figure()
        sns.set_theme(style="whitegrid")
        # This is where the actual plot gets made
        f, ax = plt.subplots(1, 2, sharey=True)
        plt.setp(ax, yticks=np.arange(0, 100 + 1, 5.0))
        ax[0] = sns.lineplot(x='Generation', y='Mean of Max', data=mean_max, ax=ax[0])
        ax[0] = sns.lineplot(x='Generation', y='Mean of Mean', data=mean_mean, ax=ax[0])
        ax[0].fill_between(mean_max['Generation'], mean_max['Mean of Max'] - std_max['Std of Max'],
                           mean_max['Mean of Max'] + std_max['Std of Max'], alpha=0.2)
        ax[0].fill_between(mean_mean['Generation'], mean_mean['Mean of Mean'] - std_mean['Std of Mean'],
                           mean_mean['Mean of Mean'] + std_mean['Std of Mean'], alpha=0.2)

        ax[0].set_title(survivor_selection)
        ax[0].legend(title='Legend', loc='lower right', labels=["Max", "StdMax", "Mean", "StdMean"])
        ax[0].set(xlabel='Generation', ylabel='Fitness')

        # Read data from file
        survivor_selection = "plus"
        mean_max, std_max, mean_mean, std_mean = self._read_data_line(survivor_selection)
        ax[1] = sns.lineplot(x='Generation', y='Mean of Max', data=mean_max, ax=ax[1])
        ax[1] = sns.lineplot(x='Generation', y='Mean of Mean', data=mean_mean, ax=ax[1])

        ax[1].fill_between(mean_max['Generation'], mean_max['Mean of Max'] - std_max['Std of Max'],
                           mean_max['Mean of Max'] + std_max['Std of Max'], alpha=0.2)
        ax[1].fill_between(mean_mean['Generation'], mean_mean['Mean of Mean'] - std_mean['Std of Mean'],
                           mean_mean['Mean of Mean'] + std_mean['Std of Mean'], alpha=0.2)

        ax[1].set_title(survivor_selection)
        ax[1].legend(title='Legend', loc='lower right', labels=["Max", "StdMax", "Mean", "StdMean"])
        ax[1].set(xlabel='Generation', ylabel='Fitness')
        # Save the plot in the plots folder
        plt.savefig(os.path.join(self.name, "line_plot.png"), dpi=300)

        # Read data from file
        survivor_selection = "comma"
        data, mean_gain = self._read_data_box(survivor_selection)
        # This is where the actual plot gets made
        ff, axx = plt.subplots(1, 2, sharey=True, figsize=(3.2, 4.8))
        c = next(palette)
        axx[0].set_title(survivor_selection)
        axx[0] = sns.boxplot(y='Gain', data=data, ax=axx[0], color=c)
        axx[0] = sns.swarmplot(y='Mean of Gain', data=mean_gain, ax=axx[0], color=".25")
        axx[0].set(xlabel='', ylabel='Individual Gain')

        # Read data from file
        survivor_selection = "plus"
        data, mean_gain = self._read_data_box(survivor_selection)
        # This is where the actual plot gets made
        c = next(palette)
        axx[1].set_title(survivor_selection)
        axx[1] = sns.boxplot(y='Gain', data=data, ax=axx[1], color=c)
        axx[1] = sns.swarmplot(y='Mean of Gain', data=mean_gain, ax=axx[1], color=".25")
        axx[1].set(xlabel='', ylabel='Individual Gain')
        # Save the plot in the plots folder
        plt.savefig(os.path.join(self.name, "box_plot.png"), dpi=300)

    def _read_data_line(self, survivor_selection):
        file_path = os.path.join(self.name, "stats_line_" + survivor_selection + ".out")
        data = pd.read_csv(file_path, names=["Generation", "Max", "Mean"])
        mean_max = data.groupby(['Generation'])['Max'].mean().reset_index(name='Mean of Max')
        std_max = data.groupby(['Generation'])['Max'].std().reset_index(name='Std of Max')
        mean_mean = data.groupby(['Generation'])['Mean'].mean().reset_index(name='Mean of Mean')
        std_mean = data.groupby(['Generation'])['Mean'].std().reset_index(name='Std of Mean')
        return mean_max, std_max, mean_mean, std_mean

    def _read_data_box(self, survivor_selection):
        file_path = os.path.join(self.name, "stats_box_" + survivor_selection + ".out")
        data = pd.read_csv(file_path, names=["Winner", "Gain"])
        mean_gain = data.groupby(['Winner'])['Gain'].mean().reset_index(name='Mean of Gain')
        return data, mean_gain
