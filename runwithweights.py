#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import pickle

# imports other libs
import numpy as np

import pandas as pd

experiment_name = 'GIVE-NAME'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 20

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="fastest",
				  enemymode="static",
				  level=2)

# test = pickle.load(open("test/output.p","rb"))
# print(test)
# print(pd.read_csv('test/output.csv'))
test = np.load('test/bestsolution.npy')

data = {'pe':[],'ee':[],'gain':[]}

# tests saved demo solutions for each enemy
en = 5

#Update the enemy
env.update_parameter('enemies',[en])

for i in range(5):
	# Load specialist controller
	# sol = np.loadtxt('solutions_demo/demo_'+str(en)+'.txt')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
	# env.play(sol)

	_,playerenergy,enemyenergy,_ = env.play(pcont=np.array(test))
	data['pe'].append(playerenergy)
	data['ee'].append(enemyenergy)
	data['gain'].append(playerenergy-enemyenergy)


pickle.dump(data, open(experiment_name+"/individualgain.p","wb"))
