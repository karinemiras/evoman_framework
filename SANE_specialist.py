import numpy as np

class SANE_Specialist():
    def __init__(self, env, gens, picklepath, logpath, total_neurons = 100, neurons_per_network = 10, n_networks = 100, mutation_sigma = 0.1):
        self.env = env
        self.total_neurons = total_neurons
        self.neurons_per_network = neurons_per_network
        self.n_networks = n_networks
        self.mutation_sigma = mutation_sigma

        self.n_inputs = env.get_num_sensors()
        self.n_bias = 1
        self.n_outputs = 5
        weights_per_neuron = self.n_inputs + self.n_bias + self.n_outputs
        self.pop = np.random.uniform(-1, 1, (total_neurons, weights_per_neuron))

        self.sane_execute(gens)

    # Create network from list of neuron indices
    def create_network(self, select):
        # Get neurons
        neurons = self.pop[select]
        # Arrange weights to work with demo_controller
        out_slice = self.n_bias + self.n_inputs
        net = np.concatenate([
            neurons[:,:self.n_bias], # hidden neuron bias
            neurons[:,self.n_bias:out_slice], # input weights
            np.zeros(self.n_outputs), # output bias
            neurons[:,out_slice:] # output weights
            ], None)
        return net

    # Create networks and run the simulation, then assign fitness to neurons
    def evaluate(self):
        fitnesses = np.zeros(self.total_neurons)
        counts = np.zeros(self.total_neurons)

        for i in range(self.n_networks):
            # Select random neurons to form a network
            select = np.random.choice(self.total_neurons, self.neurons_per_network, replace=False)
            counts[select] += 1
            net = self.create_network(select)
            # Evaluate network
            fitness, _, _, _, = self.env.play(pcont=net)
            # Add fitness to each neuron's cumulative fitness value
            fitnesses[select] += fitness
        
        # Set counts to be at least 1 to prevent division by 0
        counts = counts + (counts == 0)
        # Return average fitness of each neuron
        return fitnesses / counts

    # Tournament selection. sorted_pop should be sorted by fitness from low to high
    def tournament_selection(self, sorted_pop):
        c0 = np.random.randint(0, len(sorted_pop))
        c1 = np.random.randint(0, len(sorted_pop))
        return sorted_pop[max(c0, c1)]

    # One-point crossover of neurons' weights
    def crossover(self, p0, p1):
        s = np.random.randint(0, len(p0))
        child = p0.copy()
        child[s:] = p1[s:]
        return child

    # Mutate the whole population by adding some Gaussian noise to all weights
    def mutate_all(self):
        self.pop += np.random.normal(0, self.mutation_sigma, self.pop.shape)

    # Perform parent selection, crossover and mutation
    def new_gen(self, fitnesses):
        sorted_pop = self.pop[np.argsort(fitnesses)]
        best_half = sorted_pop[self.total_neurons//2:]

        offspring = list(best_half)
        while len(offspring) < self.total_neurons:
            p0 = self.tournament_selection(best_half)
            p1 = self.tournament_selection(best_half)
            child = self.crossover(p0, p1)
            offspring.append(child)

        self.pop = np.array(offspring)
        self.mutate_all()

    # Run the GA
    def sane_execute(self, n_gens):
        for gen in range(n_gens):
            fitnesses = self.evaluate()
            self.new_gen(fitnesses)
            print(f"Generation {gen + 1} done. Min: {fitnesses.min():.2f}, max: {fitnesses.max():.2f}, avg: {fitnesses.mean():.2f}")
