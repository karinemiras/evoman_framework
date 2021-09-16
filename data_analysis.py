# -*- coding: utf-8 -*-
"""
Spyder Editor

Models martinez 2012
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

data = []
with open('individual_demo/results.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        data.append(row)