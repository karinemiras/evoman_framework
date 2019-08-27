from controller import Controller
import numpy as np

def sigmoid_activation(x):
	return 1/(1+np.exp(-x))

# implements controller structure for player
class player_controller(Controller):
	def __init__(self):
		# Number of hidden neurons
		self.n_hidden = [10]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1
			weights1 = controller[:len(inputs)*self.n_hidden[0]].reshape((len(inputs),self.n_hidden[0]))
			bias1 = controller[len(inputs)*self.n_hidden[0]:len(inputs)*self.n_hidden[0] + self.n_hidden[0]].reshape(1,self.n_hidden[0])

			# Outputting activated first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			weights2 = controller[len(inputs)*self.n_hidden[0]+self.n_hidden[0]:-5].reshape((self.n_hidden[0],5))
			bias2 = controller[-5:].reshape(1,5)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			weights = controller[:len(inputs)*5].reshape((len(inputs), 5))
			bias = controller[len(inputs)*5:].reshape(1, 5)

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
	def __init__(self):
		# Number of hidden neurons
		self.n_hidden = [10]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1
			weights1 = controller[:len(inputs)*self.n_hidden[0]].reshape((len(inputs),self.n_hidden[0]))
			bias1 = controller[len(inputs)*self.n_hidden[0]:len(inputs)*self.n_hidden[0] + self.n_hidden[0]].reshape(1,self.n_hidden[0])

			# Outputting activated first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			# Even though the enemy only has 4 attacks 5 outputs are used so that the same network structure as the player controller can be used
			weights2 = controller[len(inputs)*self.n_hidden[0]+self.n_hidden[0]:-5].reshape((self.n_hidden[0],5))
			bias2 = controller[-5:].reshape(1,5)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			weights = controller[:len(inputs)*5].reshape((len(inputs), 5))
			bias = controller[len(inputs)*5:].reshape(1, 5)

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
