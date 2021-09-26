import numpy as np

def crossover(parent_1, parent_2, crossover_rate):
    # randomly select whether to perform crossover
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent_1))
        off_spring_1= np.append(parent_1[:crossover_point], parent_2[crossover_point:])
        off_spring_2 = np.append(parent_2[:crossover_point], parent_1[crossover_point:])
    else:
        off_spring_1, off_spring_2 = parent_1.copy(), parent_2.copy()
    
    return [off_spring_1, off_spring_2]

print(np.random.rand())
p1 = np.array([1,2,3,4,5])
p2 = np.array([6,7,8,9,10])

print(crossover(p1,p2, 0.9))