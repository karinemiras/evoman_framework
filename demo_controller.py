# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))


# implements controller structure for player
class player_controller(Controller):
	def __init__(self, _n_hidden):
		self.n_hidden = [_n_hidden]

	def set(self,controller, n_inputs):
		# Number of hidden neurons

		if self.n_hidden[0] > 0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
			self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))

			# Outputs activation first layer.


			# Preparing the weights and biases from the controller of layer 2
			self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
			self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(self.weights2)+ self.bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

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


# implements controller structure for enemy
class enemy_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
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
