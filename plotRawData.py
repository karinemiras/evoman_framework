import os
import sys
import numpy as np
import matplotlib.pyplot as pyplot
import csv

def main():
    root_folder = sys.argv[1]
    algorithm = sys.argv[2]
    run = sys.argv[3]
    enemy = sys.argv[4]
    data_type = sys.argv[5]
    graph_name = sys.argv[6]
    y_label = sys.argv[7]
    x_label = sys.argv[8]

    path = f'{root_folder}/{algorithm}/{run}/{enemy}/raw-data'

    if data_type != 'rewards' and data_type != 'fitness' and data_type != 'wins':
        sys.exit()

    files = os.listdir(path)
    print(files)
    for file in files:
        label = file
        full_path = f'{path}/{file}/{data_type}.csv'
        f = open(full_path, newline='\n')
        data = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
        episodes            = np.array([int(a[0]) for a in data])
        evaluation_episodes = np.array([int(a[1]) for a in data])
        wins                = np.array([a[3:int(a[1])] for a in data])
        means               = np.array([np.mean(a) for a in wins])
        smoothened_means    = np.array(average(means, 2))
        error               = np.array([np.sqrt(np.sum([np.square(b - np.mean(a)) for b in a])/len(a)) for a in wins])
        pyplot.plot(
            episodes,
            smoothened_means,
            label=label,
        )
        if data_type == 'rewards' or data_type == 'fitness':
            pyplot.fill_between(
                episodes,
                smoothened_means-average(error, 2),
                smoothened_means+average(error, 2),
                alpha=0.5
            )
    pyplot.title(graph_name)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.grid(True)
    if data_type == 'rewards' or data_type == 'fitness':
        pyplot.ylim(-100, 100)
    else :
        pyplot.ylim(0, 1)
    pyplot.legend()
    if not os.path.exists(f'{root_folder}/{algorithm}/{run}/plots/{data_type}'):
        os.makedirs(f'{root_folder}/{algorithm}/{run}/plots/{data_type}')
    pyplot.savefig(f'{root_folder}/{algorithm}/{run}/plots/{data_type}/{graph_name}.jpg')
    # pyplot.show()

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
