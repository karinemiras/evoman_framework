#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from scipy.stats import normaltest
from scipy.stats import ttest_ind


# In[14]:


box = pd.read_csv("boxplotdata.csv")
line = pd.read_csv("lineplotdata.csv")
box.head()


# In[28]:


print(ttest_ind(box['EAuni_enemy2'], box['EAgaus_enemy2']))


# In[26]:


print(ttest_ind(box['EAuni_enemy5'], box['EAgaus_enemy5']))


# In[27]:


print(ttest_ind(box['EAuni_enemy6'], box['EAgaus_enemy6']))

