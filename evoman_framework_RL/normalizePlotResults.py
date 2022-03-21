import sys
import numpy as np
import matplotlib.pyplot as pyplot
import csv

def main():
    filename = sys.argv[1]
    graphname = sys.argv[2]
    data_type = "r"
    ylabel = "Rewards"
    xlabel = "Steps"
    with open(filename+'_rewards.csv', newline='\n') as reward_file:
        with open(filename+'_lengths.csv', newline='\n') as lengths_file:
            rewards = list(csv.reader(reward_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
            rewards = rewards[:-1]
            # lengths = list(csv.reader(lengths_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
            labels = []
            for rew in rewards:
                labels.append(rew.pop(0))
                rew.pop(0)
            rewards = np.array(rewards)
            highest = float('-inf')
            for i in range(len(rewards)):
                rewards[i] = np.array(average(rewards[i], 2))
                if np.max(rewards[i]) > highest:
                    highest = np.max(rewards[i])
                if -np.min(rewards[i]) > highest:
                    highest = -np.min(rewards[i])

            rewards = rewards/highest
            for i in range(len(rewards)):
                # lens = lengths[i]
                # lens.pop(0)
                # lens.pop(0)
                # print(len(lens))

                rew = rewards[i]
                label = labels[i]
                # winratio = rew.pop(0)
                # label += ' (' + str(winratio) + ')'
                # pyplot.plot(lengths_to_indexes(lens), rew, label=label)
                # lengths_to_indexes(lens)

                pyplot.plot(rew, label=label)
                # pyplot.plot(lens, minimum(rew, 50), label=label + ' min')
                # pyplot.plot(lens, maximum(rew, 50), label=label + ' max')

            MSQ = []
            for i in range(len(rewards[i])):
                SQ = []
                highest = float('-inf')
                for rew in rewards:
                    if rew[i] > highest:
                        highest = rew[i]
                    if -rew[i] > highest:
                        highest = rew[i]
                for rew in rewards:
                    SQ.append(np.abs((1-(rew[i]/highest))))
                MSQ.append(np.mean(SQ))
            pyplot.plot(MSQ, label="MSQ", color = '#000000')
            pyplot.title(graphname)
            pyplot.xlabel(xlabel)
            pyplot.ylabel(ylabel)
            pyplot.grid(True)
            if (data_type == '%'):
                pyplot.ylim(-1,1)
            # pyplot.legend()
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
