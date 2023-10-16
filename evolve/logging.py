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
    def __init__(self, name):
        self.name = name
        self.generations = np.array([])
        self.best_fitness = np.array([])
        self.mean_fitness = np.array([])
        self.winner = np.array([])
        self.gain = np.array([])
        self.stats_line = []
        self.stats_box = []

    def gather_line(self, pop_fit, gen, enemies):
        self.generations = np.concatenate([self.generations, [gen]])
        self.best_fitness = np.concatenate([self.best_fitness, [np.max(pop_fit)]])
        self.mean_fitness = np.concatenate([self.mean_fitness, [np.mean(pop_fit)]])

        self.stats_line = np.stack([self.generations, self.best_fitness, self.mean_fitness])

        np.savetxt(f"{self.name}/stats_line_" + str(enemies) + ".out", self.stats_line.T, delimiter=',',
                   fmt='%1.6f')

    def gather_box(self, winner, gain, enemies):
        self.winner = np.concatenate([self.winner, [winner]])
        self.gain = np.concatenate([self.gain, [gain]])

        self.stats_box = np.stack([self.winner, self.gain])

        np.savetxt(f"{self.name}/stats_box_" + str(enemies) + ".out", self.stats_box.T, delimiter=',',
                   fmt='%1.6f')

    def draw_plots(self, enemies):
        # Read data from file
        mean_max, std_max, mean_mean, std_mean = self._read_data_line(enemies)
        # Set up line plot
        sns.set(font_scale=2, style="whitegrid")
        palette = itertools.cycle(sns.color_palette())
        plt.figure()


        fig_line, ax_line = plt.subplots()

        # This is where the actual plot gets made
        plt.setp(ax_line, yticks=np.arange(0, 100 + 1, 10.0))
        ax_line = sns.lineplot(x='Generation', y='Mean of Max', data=mean_max , label='big')
        ax_line = sns.lineplot(x='Generation', y='Mean of Mean', data=mean_mean, label='big')
        ax_line.set_title(self.name.split("_")[-1] + " " + str(enemies))
        plt.xlabel('Gen', fontsize=20, labelpad=-4)
        plt.ylabel('Fitness', fontsize=20, labelpad=4)
        plt.tick_params(labelsize=20)
        ax_line.fill_between(mean_max['Generation'], mean_max['Mean of Max'] - std_max['Std of Max'],
                         mean_max['Mean of Max'] + std_max['Std of Max'], alpha=0.2)
        ax_line.fill_between(mean_mean['Generation'], mean_mean['Mean of Mean'] - std_mean['Std of Mean'],
                         mean_mean['Mean of Mean'] + std_mean['Std of Mean'], alpha=0.2)


        ax_line.legend(title='Legend', loc='lower right', labels=["Max", "StdMax", "Mean", "StdMean"])
        # Save the plot in the plots folder
        fig_line.savefig(os.path.join(self.name, "line_plot_" + str(enemies) + ".png"), dpi=300)



        # Read data from file
        data, mean_gain = self._read_data_box(enemies)
        # Set up box plot
        sns.set(rc={'figure.figsize': (3.2, 4.8)}, font_scale=2, style="whitegrid")
        palette = itertools.cycle(sns.color_palette())
        plt.figure()
        # This is where the actual plot gets made
        fig_box, ax_box = plt.subplots()
        sns.set_theme(style="whitegrid")
        sns.set(font_scale=2)
        c = next(palette)
        plt.ylabel('Average Gain', fontsize=20, labelpad=4)
        ax_box = sns.boxplot(y='Gain', data=data, color=c)
        ax_box.set_title(self.name.split("_")[-1] + " " + str(enemies))
        plt.tick_params(labelleft=False, labelright= True, labelsize=20, pad=-4)
        ax_box = sns.swarmplot(y='Mean of Gain', data=mean_gain, color=".25")

        fig_box.savefig(os.path.join(self.name, "box_plot_" + str(enemies) + ".png"), dpi=300)

    def _read_data_line(self, enemies):
        file_path = os.path.join(self.name, "stats_line_" + str(enemies) + ".out")
        data = pd.read_csv(file_path, names=["Generation", "Max", "Mean"])
        mean_max = data.groupby(['Generation'])['Max'].mean().reset_index(name='Mean of Max')
        std_max = data.groupby(['Generation'])['Max'].std().reset_index(name='Std of Max')
        mean_mean = data.groupby(['Generation'])['Mean'].mean().reset_index(name='Mean of Mean')
        std_mean = data.groupby(['Generation'])['Mean'].std().reset_index(name='Std of Mean')
        return mean_max, std_max, mean_mean, std_mean

    def _read_data_box(self, enemies):
        file_path = os.path.join(self.name, "stats_box_" + str(enemies) + ".out")
        data = pd.read_csv(file_path, names=["Winner", "Gain"])
        mean_gain = data.groupby(['Winner'])['Gain'].mean().reset_index(name='Mean of Gain')
        return data, mean_gain
