import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np


# To open the file

for i in range(1, 10):
    data = "exp" + str(i) + ".p"
    test = pickle.load(open(data,"rb"))
    gen = np.linspace(0,len(test["avg"]),len(test["avg"]))

    # Plottign stuff
    # plt.plot(gen,test['avg'])
    # plt.fill_between(gen, test['avg'] - test['std'], test['avg'] + test['std'],
    #                  color='gray', alpha=0.2)
    plt.show()

    plt.plot(gen,test["max"])
    print("for experiment", str(i), "the max fitness is", test["max"][-1])
    plt.show()
