import numpy as np


def init_weights(in_features, out_features):
    k = 1.0 / in_features
    sqk = np.sqrt(k)
    return np.random.uniform(-sqk, sqk, (out_features, in_features))


def shitty_normalize_input(x):
    return (x - min(x)) / float((max(x) - min(x)))


def normalize_input(x):
    min = np.min(x)
    max = np.max(x)
    x_norm = x - min / (max - min)
    return x_norm


def sigmoid(x):
    y = np.zeros_like(x)
    c1 = x < -50.0
    y[c1] = 0
    c2 = (x < 0.0) & (x >= -50.0)
    y[c2] = np.exp(x[c2]) / (1 + np.exp(x[c2]))
    c3 = (x >= 0.0) & (x <= 50.0)
    y[c3] = np.exp(x[c3]) / (1 + np.exp(x[c3]))
    c4 = x > 50.0
    y[c4] = 1
    return y


def relu(x):
    return np.maximum(0, x)
