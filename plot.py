import matplotlib.pyplot as plt
import pickle

# To open the file
test = pickle.load(open("output.p","rb"))

# Plottign stuff
plt.plot(test['avg'])
plt.show()