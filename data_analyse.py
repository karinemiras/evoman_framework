# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:51:24 2021

@author: pjotr
"""

import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS

folder = 'test_run'
run = 0
data = pd.read_csv(f'{folder}/full_data_index_{run}.csv')

sequences = []
with open(f'{folder}/full_data_{run}.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        sequences.append(row)

data.insert(5, 'seq', sequences)
data_sorted = data.sort_values(by=['fitness'], ascending=False)
data_subset = data_sorted.iloc[0:50]

matrix = []
for m in data_subset['seq']:
    matrix.append(m)
matrix = np.array(matrix)

dist_matrix = distance_matrix(matrix, matrix)/np.sqrt(4*len(matrix[0]))

plt.figure(dpi=300)
plt.imshow(dist_matrix)

for i in range(len(dist_matrix)):
    plt.text(len(dist_matrix)+0.1, i+0.3, data_subset['p_health'].iloc[i], size=5)

plt.colorbar()
plt.title('joeeeee')
plt.show()


###### Multidimensional scaling
scaled_data = MDS(dissimilarity='precomputed')
scaled_data = scaled_data.fit_transform(dist_matrix)
norm = mpl.colors.Normalize(vmin=80, vmax=100)
plt.scatter(scaled_data[:,0], scaled_data[:,1], 
            c=norm(data_subset['fitness']), cmap='viridis', alpha=0.6)
plt.colorbar()
plt.show()