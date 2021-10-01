import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np
import csv

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name="EA2_gains",
				  player_controller=player_controller(n_hidden_neurons),
                  randomini='yes',
                  savelogs='no',     
                  clockprec='low',
                  playermode="ai",
                  speed="fastest",
                  enemymode="static",
                  contacthurt="player",
				  level=2)

for en in [2, 6, 8]:
    #Update the enemy
    env.update_parameter('enemies',[en])

    mean_gains = []
    # Measure best of every run
    for run in range(10):
        # Average over 5 times
        gains = []
        for i in range(5):
            sol = np.load('EA2_improved/enemy-{}/run-{}/improved-solution-{}.npy'.format(en, run, 100))
            print("Run {}, iteration {}".format(run, i))
            f, p, e, t = env.play(sol)
            gains.append(p - e)
        mean_gains.append(np.mean(gains))

    filepath = os.path.join('EA2_improved', 'mean-gains-{}.csv'.format(en))
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(mean_gains)
        f.close()
