import os
import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pyplot
import csv

def main():
    root = sys.argv[1]
    ini = sys.argv[2]
    enemy = sys.argv[3]
    graph_name = sys.argv[4]
    y_label = sys.argv[5]
    x_label = sys.argv[6]

    root_folder = f'{root}/{ini}'

    algorithms = os.listdir(root_folder)
    box_names = []
    boxes = []
    for alg in algorithms:
        alg_path = f'{root_folder}/{alg}'
        runs = os.listdir(alg_path)
        means_array = []
        for run in runs:
            try:
                data_path = f'{alg_path}/{run}/{enemy}/raw-data/({"{0.5}, {0.5}"})/rewards.csv'
                f = open(data_path, newline='\n')
                data = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
                values = np.array([a[3:int(a[1])] for a in data])
                means_array.append(np.mean(values[-1]))
            except:
                print("No full data for run")
        box_names.append(alg)
        boxes.append(means_array)

    pyplot.boxplot(x=boxes, labels=box_names)
    for box in boxes:
        print(len(box))
    # height = 100
    # for i in range(1, 3):
    #     for j in range(i, 3):
    #         height += 10
    #         pyplot.hlines(height, 1+i, 4-(j-i))
    #         pyplot.vlines(1+i, height-4, height)
    #         pyplot.vlines(4-(j-i), height-3, height)
    #         pvalue = stats.ranksums(boxes[i], boxes[3-(j-i)]).pvalue
    #         print(pvalue)
    #         if pvalue < 0.001:
    #             label = "***"
    #         elif pvalue < 0.01:
    #             label = "**"
    #         elif pvalue < 0.05:
    #             label = "*"
    #         else:
    #             label = "N.S."
    #
    #         pyplot.annotate(label, ((1+i + 4-(j-i))/2, height))

    pyplot.title(graph_name)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    # if not os.path.exists(f'{root}/box-plots/{ini}'):
    #     os.makedirs(f'{root}/box-plots/{ini}')
    # pyplot.savefig(f'{root}/box-plots/{ini}/{graph_name}.jpg')
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
