import numpy as np

from evoman.controller import Controller
from evolve.util import sigmoid, relu, init_weights, normalize_input
from math import prod

class NNController(Controller):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def set(self, net, _):
        self.net = net

    def control(self, params, net = None):
        if net is None:
            nn_out = self.net(params)
        else:
            nn_out = net(params)
        return [int(elem > self.threshold) for elem in nn_out]


class NeuralNetwork:
    def __init__(self, input, hidden, output, activation="sigmoid"):
        self.w1 = init_weights(input, hidden)
        self.b1 = init_weights(1, hidden).flatten()
        self.w2 = init_weights(hidden, output)
        self.b2 = init_weights(1, output).flatten()
        self._params_list = self._convert_params_to_list()

        if activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "relu":
            self.activation = relu
        else:
            raise ValueError("Unknown activation passed as init argument")
        
    def __call__(self, x):
        x = normalize_input(x)
        x = np.dot(self.w1, x) + self.b1
        x = self.activation(x)
        x = np.dot(self.w2, x) + self.b2
        x = self.activation(x)
        return x
    
    def load_weights(self, filepath):
        with open(filepath, 'r') as file:
            weights = file.read()[1:-1].split(',')
            weights = [float(w) for w in weights]
            self._update_params(weights)
    
    def save_weights(self, filepath):
        weights = self._params_list
        with open(filepath, 'w') as out:
            out.write(str(weights))
    
    def __len__(self):
        return len(self._params_list)
    
    def __setitem__(self, index, item):
        self._params_list[index] = item
        self._update_params(self._params_list)

    def __getitem__(self, index):
        return self._params_list[index]

    def _convert_params_to_list(self):
        params_list = []
        for param in self._get_params():
            params_list += param.reshape(-1).tolist()
        return params_list
    
    def _update_params(self, new_params_list):
        shapes = [param.shape for param in self._get_params()]
        new_params = []
        begin = 0 
        for shape in shapes:
            length = prod(shape)
            param = new_params_list[begin: begin + length]
            param_array = np.array(param).reshape(shape)
            new_params.append(param_array)
            begin += length
        self.w1 = new_params[0]
        self.b1 = new_params[1]
        self.w2 = new_params[2]
        self.b2 = new_params[3]

    def _get_params(self):
        return [self.w1, self.b1, self.w2, self.b2]
