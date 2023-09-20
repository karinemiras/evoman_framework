import sys, os
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np

experiment_name = 'controller_specialist_sandor'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


en = 5
env.update_parameter('enemies',[en])
sol = np.loadtxt('neat-controller/winner_genome5.txt')
print('\n Loading solution vector for enemy '+str(en)+' \n')
env.play(sol)
