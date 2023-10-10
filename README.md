Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI


## Setup
To install requirements type in command:
`pip install -r requirements.txt`

If you want to train NN with DEAP network run:
`python controller_specialist_deap.py`

or 

`python controller_generalist_deap.py`

## Hyperparameters
In the file `config.yaml` there's a set of parameters you can tune. The config utilizes `hydra` library.

## Inference
When you trained neural net from the above script you can see it in action with the following command:
`python controller_specialist_deap.py`

## DEAP
All files ending with "deap"

## NEAT
All files in the folder "neat"

## Optuna
In order to run training with automatic hyperparameter search, you can run one of the following commands:

`python controller_specialist_deap.py --multirun`

or 

`python controller_generalist_deap.py --multirun`

The results of your experiments can be found in directory `multirun/{timestamp}/optimization_results.yaml`

## Legend:
"optimization_" files are used to find the best solution

"contoller_..." these files run the soulution found by "optimization_..." files

(for Task 1 consider just the files containing "specialist" in their name

The given neural network is in the file "demo_controller.py"

All files containing "dummy" are draft files from where we can start to implement our own solutions

The neural net that was implemented by Jacob is in the folder "evolve"