################################
# NEAT ALGORITHM APPLIED TO EVOMAN FRAMEWORK
# This document is the main and is used to tune the parameters of the NEAT algorithm.
# It is also used to show figures.
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')

# imports other libs
import numpy as np
from environment import Environment
import apply_NEAT
import csv
import yaml
import pandas as pd
import math
import itertools
import figures_NEAT_tuning
from specialist_controller import NEAT_Controls

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Load the configuration file
with open("config_main.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

name_experiment = cfg["decision_variables"]["name_experiment"]
tune_parameters = cfg["decision_variables"]["tune_parameters"] # boolean which decides to tune parameters
enemy = cfg["experiment_parameters"]["enemy"]
generations = cfg["experiment_parameters"]["generations"]
apply_optimization = cfg["decision_variables"]["apply_optimization"]
show_figures = cfg["decision_variables"]["show_figures"]
save_figures = cfg["decision_variables"]["save_figures"]


if tune_parameters:
    tuning_parameters = cfg["tuning_parameters"]

run_nr = cfg["experiment_parameters"]["number_of_runs"]  # number of runs per




if not os.path.exists(name_experiment):
    os.makedirs(name_experiment)



def change_config_file(config, parameter_names, parameter_values):
    file = open(config, "r")
    list_of_lines = file.readlines()

    for jdx, parameter_name in enumerate(parameter_names):
        line = [idx for idx, s in enumerate(list_of_lines) if parameter_name in s][0]
        string = list_of_lines[line]
        name = string[:string.index("=")]
        list_of_lines[line] = name+"= "+str(parameter_values[jdx])+"\n"

    file = open(config, "w")
    file.writelines(list_of_lines)
    file.close()

    return config

def integer_options(series):
    l_bound = series["l_bound"]
    u_bound = series["u_bound"]
    num_var = series["num_variations"]

    options = list(range(math.ceil(l_bound), math.floor(u_bound)+1))

    step_size = int(len(options)/num_var)

    final_options = np.zeros(num_var)

    k = 0
    for i in range(num_var-1):
        final_options[i] = options[k]
        k+=step_size

    final_options[(num_var-1)] = options[(len(options)-1)]

    return np.array(options)


def float_options(series):
    l_bound = series["l_bound"]
    u_bound = series["u_bound"]
    num_var = series["num_variations"]

    step_size = (u_bound-l_bound)/(num_var-1)

    options = np.arange(l_bound, (u_bound+step_size), step_size)

    return np.array(options)

def all_combinations(tune_options):
    all_combinations = list(itertools.product(*tune_options))

    return all_combinations



def make_parameter_matrix():
    df = pd.DataFrame([])
    tune_options = []

    for param in tuning_parameters:
        df.index = tuning_parameters[param]
        df[param] = tuning_parameters[param].values()

        if df.loc["type",param] == "integer":
            tune_options.append(integer_options(df[param]))

        if df.loc["type",param] == "float":
            tune_options.append(float_options(df[param]))

    parameter_options = all_combinations(tune_options)
    return parameter_options, df

def run_experiment(env,  config, parameter_options = None, parameter_names=None):
    if tune_parameters:
        for parameters in parameter_options:
            config = change_config_file(config, parameter_names, parameters)
            for run in range(run_nr):
                apply_NEAT.run(env, generations, config, str(parameters)+ "_run_"+str(run), name_experiment)

    else:
        for run in range(run_nr):
            apply_NEAT.run(env, generations, config, run, name_experiment)

def main(config):

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=name_experiment,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=NEAT_Controls(),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      randomini="yes")

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state

    if tune_parameters:
        parameter_options, parameter_descriptions = make_parameter_matrix()
        parameter_names = parameter_descriptions.loc["name", :]

    if apply_optimization:
        run_experiment(env, config, parameter_options, parameter_names)

    if save_figures or show_figures:
        figures_NEAT_tuning.make_figures(save_figures, show_figures, name_experiment, parameter_options, parameter_names,run_nr, enemy )

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_NEAT_path = os.path.join(local_dir, 'config-feedforward')
    main(config_NEAT_path)