####################
# Use this file to run an optimization algorithm
# Specify the number of runs if you want to run more than one individual optimization

# User arguments: 
#   1) algorithm (standard or SANE)
#   2) enemy number (1-8)
#   3) number of optimizations to run (1-10) (default: 1)

# The framework creates a directory based on 'experiment_name' where it saves the evoman logs
# This directory should also be used by the optimization algorithm to store fitness data
# Each run gets its own folder run# inside the original experiment_name location

####################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller


# Read command line arguments
if not(len(sys.argv) == 3 or len(sys.argv) == 4):
    sys.exit("Error: please specify:\n1) the algorithm (standard or SANE)\n2) the enemy (1-8)\n3) number of individual optimizations (1-10) (default:1)")

# First argument must indicate the algorithm - 'standard' or 'SANE'
algorithm = sys.argv[1]

if algorithm == 'sane': 
    algorithm = 'SANE'
#print("First argument: ", algorithm)

if not(algorithm == 'standard' or algorithm == 'SANE'):
    sys.exit("Error: please specify the algorithm using 'standard' or 'SANE'.")

# Second argument must specify the enemy to be trained on - integer from 1 - 8
try:
    enemy = int(sys.argv[2])
except TypeError:
    sys.exit("Error: please specify the enemy using an integer from 1 to 8.")
    
#print("Second argument: ", enemy)

if not(enemy > 0 and enemy < 9):
    sys.exit("Error: please specify the enemy using an integer from 1 to 8.")

# Third argument must indicate the number of optimizations to run (default: 1)
if len(sys.argv) == 4:
    try:
        runs = int(sys.argv[3])
    except TypeError:
        sys.exit("Error: please specify how many optimizations you want to run (1-10). Default: 1")
    
    #print("Third argument: ", runs)
        
    if not(runs > 0 and runs < 11):
        sys.exit("Error: please specify how many optimizations you want to run (1-10). Default: 1")
else:
    runs = 1

print(runs, "experiment(s) will be run with the following settings:\nAlgorithm: ", algorithm, "\nEnemy: ", enemy)

# Define experiment name
# Default experiment name. Comment to specify your own. Cleanest to start with optimizations/
experiment_name = "optimizations/" + algorithm + "_e" + str(enemy)
#experiment_name = "optimizations/[insert name here]"
print("Logs will be saved at:", experiment_name)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
# Initialize environment
enemies = [enemy]
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  playermode="ai",
                  #player_controller= insert desired controller here,            
                  enemymode="static",
                  level=2,
                  speed="fastest")

# Run the optimizations the specified number of times
# Each run gets their own folder run# inside the original folder
for it in range(1,runs+1):
    run_name = experiment_name + "/run" + str(it)
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    env.update_parameter("experiment_name", run_name)
    # do things
    # Run the algorithm

# create instance of algorithm class
# start the optimization and pass the env object to it so they can test with env.play()?



