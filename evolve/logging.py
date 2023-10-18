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


def _draw_line_plot(directory1, directory2, enemies):
    # Read data from directories
    mean_max1, std_max1, mean_mean1, std_mean1 = _read_data_line(directory1, enemies)
    mean_max2, std_max2, mean_mean2, std_mean2 = _read_data_line(directory2, enemies)
    # Set up line plot
    sns.set(rc={'figure.figsize': (9.6, 4.8)}, font_scale=2, style="whitegrid")
    palette = itertools.cycle(sns.color_palette("bright"))
    c1b = next(palette)
    c2b = next(palette)
    palette = itertools.cycle(sns.color_palette("pastel"))
    c1p = next(palette)
    c2p = next(palette)
    plt.figure()

    # This is where the actual plot gets made
    plt.xlabel('Generation', fontsize=20, labelpad=-4)
    plt.ylabel('Fitness', fontsize=20, labelpad=-4)
    plt.tick_params(labelsize=20)

    plt.title("Enemies " + str(enemies), fontsize=20, pad=15)
    ax1 = sns.lineplot(x='Generation', y='Mean of Max', data=mean_max1, color=c1b, linewidth = 3, label="Island Max")
    sns.lineplot(x='Generation', y='Mean of Mean', data=mean_mean1, color=c1p, linewidth = 3, label="Island Mean")
    plt.fill_between(mean_max1['Generation'], mean_max1['Mean of Max'] - std_max1['Std of Max'],
                         mean_max1['Mean of Max'] + std_max1['Std of Max'], alpha=0.2, color=c1b)
    plt.fill_between(mean_mean1['Generation'], mean_mean1['Mean of Mean'] - std_mean1['Std of Mean'],
                         mean_mean1['Mean of Mean'] + std_mean1['Std of Mean'], alpha=0.2, color=c1p)



    sns.lineplot(x='Generation', y='Mean of Max', data=mean_max2, color=c2b, linewidth = 3, label="PSO Max")
    sns.lineplot(x='Generation', y='Mean of Mean', data=mean_mean2, color=c2p, linewidth = 3, label="PSO Mean")
    plt.fill_between(mean_max2['Generation'], mean_max2['Mean of Max'] - std_max2['Std of Max'],
                            mean_max2['Mean of Max'] + std_max2['Std of Max'], alpha=0.2, color=c2b)
    plt.fill_between(mean_mean2['Generation'], mean_mean2['Mean of Mean'] - std_mean2['Std of Mean'],
                            mean_mean2['Mean of Mean'] + std_mean2['Std of Mean'], alpha=0.2, color=c2p)

    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.setp(ax1, yticks=np.arange(0, 100 + 1, 10.0))


    # Save the plot in the plots folder
    if not os.path.exists("../optimization_generalist_plots"):
        os.mkdir("optimization_generalist_plots")
    plt.savefig(("../optimization_generalist_plots/line_plot_" + str(enemies) + ".png"), dpi=300)


def _draw_box_plots(directory1, directory2, enemies1, enemies2):
    # Read data from directories
    data1, data_mean1 = _read_data_box(directory1, enemies1)
    data2, data_mean2 = _read_data_box(directory2, enemies1)
    data3, data_mean3 = _read_data_box(directory1, enemies2)
    data4, data_mean4 = _read_data_box(directory2, enemies2)

    data = pd.DataFrame({'Gain1': data_mean1['Gain'], 'Gain2': data_mean2['Gain'], 'Gain3': data_mean3['Gain'], 'Gain4': data_mean4['Gain']})
    print(data)
    # Set up box plot
    sns.set(rc={'figure.figsize': (9.6, 4.8)}, font_scale=2, style="whitegrid")
    palette = itertools.cycle(sns.color_palette("bright"))
    c1b = next(palette)
    c2b = next(palette)
    palette = itertools.cycle(sns.color_palette("pastel"))
    c1p = next(palette)
    c2p = next(palette)
    plt.figure()

    # This is where the actual plot gets made
    plt.ylabel('Average Gain', fontsize=20, labelpad=-4)
    plt.title("Enemies " + str(enemies1) + "                      " + "Enemies" + str(enemies2), fontsize=20, pad=15)
    ax1 = sns.boxplot(data=data, palette=[c1b, c2b, c1p, c2p], linewidth = 3)
    #change name of xticks
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['Island', 'PSO', 'Island', 'PSO'])
    # Save the plot in the plots folder
    if not os.path.exists("optimization_generalist_plots"):
        os.mkdir("optimization_generalist_plots")
    plt.savefig(os.path.join("../optimization_generalist_plots", "both_box_plots.png"), dpi=300)


def _read_data_line(directory, enemies):
    file_path = os.path.join(directory, "stats_line_" + str(enemies) + ".out")
    data = pd.read_csv("../" + file_path, names=["Generation", "Max", "Mean"])
    mean_max = data.groupby(['Generation'])['Max'].mean().reset_index(name='Mean of Max')
    std_max = data.groupby(['Generation'])['Max'].std().reset_index(name='Std of Max')
    mean_mean = data.groupby(['Generation'])['Mean'].mean().reset_index(name='Mean of Mean')
    std_mean = data.groupby(['Generation'])['Mean'].std().reset_index(name='Std of Mean')
    return mean_max, std_max, mean_mean, std_mean


def _read_data_box(directory, enemies):
    file_path = os.path.join(directory, "stats_box_" + str(enemies) + ".out")
    data = pd.read_csv("../" + file_path, names=["Winner", "Gain"])
    data_mean = data.groupby(['Winner'])["Gain"].mean().reset_index(name='Gain')
    return data, data_mean


if __name__ == "__main__":
    _draw_line_plot("optimization_generalist_island", "optimization_generalist_pso", [2, 3, 5, 8])
    _draw_line_plot("optimization_generalist_island", "optimization_generalist_pso", [1, 5, 6])
    _draw_box_plots("optimization_generalist_island", "optimization_generalist_pso", [2, 3, 5, 8], [1, 5, 6])
