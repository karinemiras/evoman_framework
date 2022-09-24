import neat
import os
from controller import Controller
import numpy as np

def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))

# player controller structure using NEAT
class NeatController(Controller):
    def __init__(self):
        neat_dir = os.path.dirname(__file__)
        neat_config = os.path.join(neat_dir, "NEAT-config.txt")
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    neat_config)

        self.config = config

    def control(self, input_sensors, genome):
        network = neat.nn.FeedForwardNetwork.create(genome, self.config)
        output = network.activate(input_sensors)

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

class enemy_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        if self.n_hidden[0]>0:
            bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs), self.n_hidden[0]))
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)
            bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
            weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        if output[0] > 0.5:
            attack1 = 1
        else:
            attack1 = 0

        if output[1] > 0.5:
            attack2 = 1
        else:
            attack2 = 0

        if output[2] > 0.5:
            attack3 = 1
        else:
            attack3 = 0

        if output[3] > 0.5:
            attack4 = 1
        else:
            attack4 = 0
        return [attack1, attack2, attack3, attack4]