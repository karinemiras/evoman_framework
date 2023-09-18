Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI


## Setup
To install requirements type in command:
`pip install -r requirements.txt`

If you want to train NN with DEAP network run:
`python neural_net_deap_example.py`

## Hyperparameters
In the file `config.yaml` there's a set of parameters you can tune. The config utilizes `hydra` library.

## Inference
When you trained neural net from the above script you can see it in action with the following command:
`python neural_net_deap_demo.py`
