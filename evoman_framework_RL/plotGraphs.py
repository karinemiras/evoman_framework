import os
import sys
import numpy as np
import matplotlib.pyplot as pyplot
import csv

def main():
    root = sys.argv[1]
    ini = sys.argv[2]
    enemy = sys.argv[3]
    data_type = sys.argv[4]
    graph_name = sys.argv[5]
    y_label = sys.argv[6]

    fig = pyplot.figure() #figsize=(7.5, 5.5)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    pyplot.subplots_adjust(top=0.85)

    root_folder = f'{root}/{ini}'

    algorithms = ["DQN", "PPO", "GA-50", "NEAT"]
    for alg in algorithms:
        alg_path =  f'{root_folder}/{alg}'
        runs = os.listdir(alg_path)
        episode_array = []
        mean_arrays = []
        for run in runs:
            try:
                data_path = f'{alg_path}/{run}/{enemy}/raw-data/({"{0.5}, {0.5}"})/{data_type}.csv'
                f = open(data_path, newline='\n')
                data = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
                episodes = np.array([int(a[0]) for a in data])
                values = np.array([a[3:int(a[1])] for a in data])
                means = np.array([np.mean(a) for a in values])
                episode_array = episodes
                mean_arrays.append(means)
            except:
                print("No full data for run")
        if alg == 'GA-50' or alg == 'GA-50-untimed' or alg == 'NEAT':
            episode_array *= 25000

        try:
            tranformed = [[mean_arrays[x][y] for x in range(len(mean_arrays))] for y in range(len(mean_arrays[0]))]
            full_mean = np.array([np.percentile(a, 50) for a in tranformed])
            q75, q25 = [np.array([np.percentile(a, 75) for a in tranformed]),
                        np.array([np.percentile(a, 25) for a in tranformed])]
            smoothened_mean = np.array(average(full_mean, 2))
            # error = np.array([np.sqrt(np.sum([np.square(b - np.mean(a)) for b in a]) / len(a)) for a in tranformed])
            ax1.plot(episode_array, smoothened_mean, label=alg)
            ax1.fill_between(
                episode_array,
                average(q25, 2),
                average(q75, 2),
                alpha=0.5
            )
        except:
            print("No full data for run")

    if data_type == 'rewards':
        pyplot.ylim(-105, 105)
    elif data_type == 'wins':
        pyplot.ylim(-0.05, 1.05)

    pyplot.title(graph_name)
    ax1.set_xlabel('timesteps')
    ax1.set_ylabel(y_label)
    ax1.grid(True)
    ax1.legend()

    locs, labels = pyplot.xticks()
    print(locs, labels)
    ax2.set_xlabel('generations')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(locs * 2.5e6)
    ax2.set_xticklabels([int(l*2.5e6/25000) for l in locs])

    if not os.path.exists(f'{root}/plots/{ini}/{data_type}'):
        os.makedirs(f'{root}/plots/{ini}/{data_type}')
    pyplot.savefig(f'{root}/plots/{ini}/{data_type}/{graph_name}.jpg')
    pyplot.show()

def lengths_to_indexes(array):
    for i in range(1, len(array)):
        array[i] = array[i] + array[i-1]

    return array

def average(array, smoothing_factor):
    average = []
    for i in range(len(array)):
        total = 0
        amount = 0
        for j in range(max(0, i-smoothing_factor), min(i+smoothing_factor, len(array) - 1)):
            total += array[j]
            amount += 1
        average.append(total/amount)
    # average.append(array.pop())

    return average


def maximum(array, smoothing_factor):
    max_values = []
    for i in range(len(array)):
        max_val = float('-inf')
        for j in range(max(0, i-smoothing_factor), min(i+smoothing_factor, len(array) - 1)):
            max_val = max(max_val, array[j])
        max_values.append(max_val)
    
    return max_values


def minimum(array, smoothing_factor):
    min_values = []
    for i in range(len(array)):
        min_val = float('inf')
        for j in range(max(0, i-smoothing_factor), min(i+smoothing_factor, len(array) - 1)):
            min_val = min(min_val, array[j])
        min_values.append(min_val)
    
    return min_values

if __name__ == "__main__":
    main()
