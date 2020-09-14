import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np


# To open the file
test = pickle.load(open("output.p","rb"))
gen = np.linspace(0,len(test["avg"]),len(test["avg"]))

# Plottign stuff
plt.plot(gen,test['avg'])
plt.fill_between(gen, test['avg'] - test['std'], test['avg'] + test['std'],
                 color='gray', alpha=0.2)
plt.show()

plt.plot(gen,test["max"])
plt.show()