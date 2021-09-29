import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

for en in [2, 6, 8]:
    data = []
    for run in range(10):
        path = os.path.join('EA2_improved', 'enemy-{}'.format(en), 'run-{}'.format(run), 'improved-results-100.csv')
        data.append(np.genfromtxt(path, delimiter=','))
    data = np.array(data)

    # (mean_average, mean_max)
    means = np.mean(data, axis=0)
    # (std_average, std_max)
    stds = np.std(data, axis=0)

    means_avg = means[:, 0]
    stds_avg = stds[:, 0]

    means_max = means[:, 1]
    stds_max = stds[:, 1]

    x = range(100)
    # Plot means, maxes and save figures
    fig, ax = plt.subplots()

    ax.plot(x, means_max, '-', label='Max')
    ax.fill_between(x, means_max - stds_max, means_max + stds_max, alpha=0.2)
    ax.plot(x, means_avg, '-', label='Mean')
    ax.fill_between(x, means_avg - stds_avg, means_avg + stds_avg, alpha=0.2)
    ax.legend()
    plt.savefig('EA2_improved/enemy-{}.png'.format(en))






