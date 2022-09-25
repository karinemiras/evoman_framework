####################
# Use this file to run an optimization algorithm: either NEAT or SANE
# Specify the number of runs if you want to run more than one individual optimization

# User arguments: 
#   1) algorithm (NEAT or SANE)
#   2) enemy number (1-8)
#   3) number of optimizations to run (1-10) (default: 1)

# The program will ask how many generations the algorithm should run

# The experiment name, and therefore location of the logs, is created by default as follows:
# optimizations/[algorithm]_e#_gen# where [algorithm] is NEAT or SANE, e# indicates enemy number and gen# indicates number of generations
# Inside this folder, each individual run will create a folder run# that will contain the logs for this run

# The algorithm will save its best solution in the run# folder. If you want to run this solution separately:
# Move the file to 'solutions' folder and run run_specialist.py

####################

# imports framework
from multiprocessing.sharedctypes import Value
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from NEAT_controller import NeatController
from NEAT_specialist import NEAT_Spealist


# Read command line arguments
if not(len(sys.argv) == 3 or len(sys.argv) == 4):
    sys.exit("Error: please specify:\n1) the algorithm (NEAT or SANE)\n2) the enemy (1-8)\n3) number of individual optimizations (1-10) (default:1)")

# First argument must indicate the algorithm - 'standard' or 'SANE'
algorithm = sys.argv[1]

if algorithm == 'neat': 
    algorithm = 'NEAT'
if algorithm == 'sane': 
    algorithm = 'SANE'
#print("First argument: ", algorithm)

if not(algorithm == 'NEAT' or algorithm == 'SANE'):
    sys.exit("Error: please specify the algorithm using 'NEAT' or 'SANE'.")

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



# Ask user for the number of generations
while True:
    try:
        gens = int(input("Enter the number of generations: "))
    except ValueError:
        print("Please enter a number")
    else:
        break
        
# Default experiment name. Comment to specify your own
experiment_name = "optimizations/" + algorithm + "_e" + str(enemy) + "_gen" + str(gens)
#experiment_name = "optimizations/[insert name here]"
print("Logs will be saved at:", experiment_name)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
# Initialize environment
enemies = [enemy]

# Select controller according to the algorithm
if algorithm == 'NEAT': control = NeatController()
else: control = "insert SANE controller"

env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  playermode="ai",
                  player_controller=control,            
                  enemymode="static",
                  level=2,
                  logs = "off",
                  speed="fastest")
    
# Run the optimizations the specified number of times
# Each run gets their own folder run# inside the original folder
for it in range(1,runs+1):
    print("\nSTARTING RUN:", it)
    run_name = experiment_name + "/run" + str(it)
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    env.update_parameter("experiment_name", run_name)
    
    # Create a path for the pickle file, so the algorithm knows where to save it
    picklepath = run_name + "/" + algorithm + "_e" + str(enemy) + ".pkl"
    logpath = run_name + "/data_run_1.txt"
    
    if algorithm == 'NEAT':
        optimizer = NEAT_Spealist(env, gens, picklepath, logpath)
    else:
        optimizer = "insert SANE optimizer"
    
    


