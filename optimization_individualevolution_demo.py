############################################################################### 
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm with perceptron neural network.   #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
############################################################################### 

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller

# imports other libs 
import time
import numpy as np
from math import fabs,sqrt
import glob, os


# implements controller structure 
class player_controller(Controller):



    def control(self, params,cont):

        params = (params-min(params))/float((max(params)-min(params))) # standardizes
        params = np.hstack( ([1.], params) )
        n_outs = [5] # number of output neurons (sprite actions)
        n_params = [len(params)] # number of input variables
        n_hlayers = 1	  # number of hidden layers
        n_lneurons = [50] # number of neurons in each hidden layer
        neurons_layers = []  # array of ann layers. Each position is a layer, and will contain another array in which each position will be a neuron.
        neurons_layers.append(params) # adds input neurons layer to the ann.


        weights = cont # reads weights of solution

        if n_hlayers==1:
            nh = n_params[0]*n_lneurons[0]
            weights1 = weights[:nh].reshape( (n_params[0], n_lneurons[0]) )
            output1 = 1./(1. + np.exp(-params.dot(weights1)))
            output1 = np.hstack( ([1],output1) )
            weights2 = weights[nh:].reshape( (n_lneurons[0]+1, n_outs[0]) )
            output = 1./(1. + np.exp(-output1.dot(weights2)))
        else:

            weights = weights.reshape( (n_params[0], n_outs[0]) )
            output = 1./(1. + np.exp(-params.dot(weights)))


        # takes decisions about sprite actions

        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0



        return [left, right, jump, shoot, release]


experiment_name = 'demo_individual'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype/weights for perceptron phenotype network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test
#n_vars = (env.get_num_sensors()+1)*5  # perceptron
#n_vars = (env.get_num_sensors()+1)*10 + 11*5  # multilayer with 10 neurons
n_vars = (env.get_num_sensors()+1)*50 + 51*5 # multilayer with 50 neurons
dom_u = 1
dom_l = -1
npop = 2#100
gens = 30
mutacao = 0.2
last_best = 0


# runs simulation
def simula(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def avalia(x):
    return np.array(map(lambda y: simula(env,y), x))


# tournment
def torneio(pop):
    c1 =  np.random.randint(0,pop.shape[0], 1)
    c2 =  np.random.randint(0,pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]


# limits
def limites(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


    # crossover
def cruzamento(pop):

    total_filhos = np.zeros((0,n_vars))


    for p in range(0,pop.shape[0], 2):
        p1 = torneio(pop)
        p2 = torneio(pop)

        n_filhos =   np.random.randint(1,3+1, 1)[0]
        filhos =  np.zeros( (n_filhos, n_vars) )

        for f in range(0,n_filhos):

            cross_prop = np.random.uniform(0,1)
            filhos[f] = p1*cross_prop+p2*(1-cross_prop)

            # mutation 
            for i in range(0,len(filhos[f])):
                if np.random.uniform(0 ,1)<=mutacao:
                    filhos[f][i] =   filhos[f][i]+np.random.normal(0, 1)

            filhos[f] = np.array(map(lambda y: limites(y), filhos[f]))

            total_filhos = np.vstack((total_filhos, filhos[f]))

    return total_filhos


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop,fit_pop):

    worst = int(npop/4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0,n_vars):
            pro = np.random.uniform(0,1)
            if np.random.uniform(0,1)  <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist. 
            else:
                pop[o][j] = pop[order[-1:]][0][j] # dna from best 

        fit_pop[o]=avalia([pop[o]])

    return pop,fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    avalia([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = avalia(pop)
    best = np.argmax(fit_pop)
    media = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    media = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop 
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(media,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(media,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    filhos = cruzamento(pop)  # crossover
    fit_filhos = avalia(filhos)   # evaluation
    pop = np.vstack((pop,filhos))
    fit_pop = np.append(fit_pop,fit_filhos)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(avalia(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(map(lambda y: norm(y,fit_pop_cp), fit_pop)) # avoiding negative probabilities, as fitness is ranges from negative numbers  
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    escolhidos = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    escolhidos = np.append(escolhidos[1:],best)
    pop = pop[escolhidos]
    fit_pop = fit_pop[escolhidos]


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results 
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution  
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open('neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
