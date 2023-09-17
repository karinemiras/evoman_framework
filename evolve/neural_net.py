import numpy as np

from evoman.controller import Controller
from math import prod


class NNController:
    def __init__(self, input, hidden, output, activation="sigmoid"):
        self.w1 = np.random.randn(hidden, input)
        self.b1 = np.random.randn(hidden)
        self.w2 = np.random.randn(output, hidden)
        self.b2 = np.random.randn(output)
        self._params_list = self._convert_params_to_list()

        if activation == "sigmoid":
            self.activation = self._sigmoid
        elif activation == "relu":
            self.activation = self._relu
        else:
            raise ValueError("Unknown activation passed as init argument")
        
    def control(self, x):
        x = self._normalize_input(x)
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

    # TODO: Implement
    def _normalize_input(self, x):
        # min = np.min(x)
        # max = np.max(x)
        # x_norm = x - min / (max-min)
        # return x_norm
        return x

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)
