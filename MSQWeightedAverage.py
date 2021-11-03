import sys
from map_enemy_id_to_name import id_to_name
import numpy as np
import matplotlib.pyplot as pyplot
import csv

def main():
    # filename = sys.argv[1]
    # graphname = sys.argv[2]
    data_type = "r"
    ylabel = "Rewards"
    xlabel = "Steps"
    graphname = "WeightedAverage"
    rewards = []
    MSQs = []
    labels = []
    for enemyn in range(8):
        filename = 'HitpointWeightIntermit/'+id_to_name(enemyn+1)
        with open(filename+'_rewards.csv', newline='\n') as reward_file:
            with open(filename+'_lengths.csv', newline='\n') as lengths_file:
                rewards.append(list(csv.reader(reward_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))[:-1])
                # lengths = list(csv.reader(lengths_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
                for rew in rewards[enemyn]:
                    if enemyn == 0:
                        labels.append(rew.pop(0))
                    else:
                        rew.pop(0)
                    rew.pop(0)
                rewards[enemyn] = np.array(rewards[enemyn])
                rewards_MSQ = np.array(rewards[enemyn])
                highest = float('-inf')
                for i in range(len(rewards_MSQ)):
                    rewards_MSQ[i] = np.array(average(rewards_MSQ[i], 2))
                    if np.max(rewards_MSQ[i]) > highest:
                        highest = np.max(rewards_MSQ[i])
                    if -np.min(rewards_MSQ[i]) > highest:
                        highest = -np.min(rewards_MSQ[i])

                rewards_MSQ = rewards_MSQ/highest

                MSQ = []
                for i in range(len(rewards_MSQ[i])):
                    SQ = []
                    highest = float('-inf')
                    for rew in rewards_MSQ:
                        if rew[i] > highest:
                            highest = rew[i]
                        if -rew[i] > highest:
                            highest = rew[i]
                    for rew in rewards_MSQ:
                        SQ.append(np.abs((1-(rew[i]/highest))))
                    MSQ.append(np.mean(SQ))
                MSQs.append(MSQ[-1])
                # pyplot.plot(MSQ, label="MSQ", color = '#000000')
                # pyplot.title(graphname)
                # pyplot.xlabel(xlabel)
                # pyplot.ylabel(ylabel)
                # pyplot.grid(True)
                # if (data_type == '%'):
                #     pyplot.ylim(-1,1)
                # # pyplot.legend()
                # pyplot.show()
    weightedRewards = []
    for _ in range(len(rewards[0])):
        weightedRewards.append(np.array([0 for _ in range(26)]))
    for enemyn in range(8):
        rews = rewards[enemyn]
        for i in range(len(rews)):
            weightedRewards[i] = weightedRewards[i] + ((MSQs[enemyn] * rews[i])/np.sum(MSQs))

    for i in range(len(weightedRewards)):
        rew = weightedRewards[i]
        label = labels[i]

        pyplot.plot(np.array(average(rew, 2)), label=label)
    pyplot.title(graphname)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.grid(True)
    pyplot.ylim(-100, 100)
    pyplot.legend()
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
