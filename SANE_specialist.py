import numpy as np

class SANE_Specialist():
    def __init__(self, env, gens, picklepath, logpath, cfg):
        self.env = env
        self.logpath = logpath
        self.total_neurons = int(cfg['total_neurons'])
        self.neurons_per_network = int(cfg['neurons_per_network'])
        self.n_networks = int(cfg['n_networks'])
        self.mutation_sigma = float(cfg['mutation_sigma'])
        self.survivor_rate = float(cfg['survivor_rate'])
        self.parent_rate = float(cfg['parent_rate'])

        self.n_inputs = env.get_num_sensors()
        self.n_bias = 1
        self.n_outputs = 5
        weights_per_neuron = self.n_inputs + self.n_bias + self.n_outputs
        self.pop = np.random.uniform(-1, 1, (self.total_neurons, weights_per_neuron))

        with open(logpath, 'w') as logfile:
            logfile.write("")

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
        neuron_fitnesses = np.zeros(self.total_neurons)
        network_fitnesses = np.zeros(self.n_networks)
        counts = np.zeros(self.total_neurons)

        for i in range(self.n_networks):
            # Select random neurons to form a network
            select = np.random.choice(self.total_neurons, self.neurons_per_network, replace=False)
            counts[select] += 1
            net = self.create_network(select)
            # Evaluate network
            fitness, _, _, _, = self.env.play(pcont=net)
            network_fitnesses[i] = fitness
            # Add fitness to each neuron's cumulative fitness value
            neuron_fitnesses[select] += fitness
        
        # Set counts to be at least 1 to prevent division by 0
        counts = counts + (counts == 0)
        # Return average fitness of each neuron and fitnesses of the networks
        return neuron_fitnesses / counts, network_fitnesses

    # Tournament selection with k=2. sorted_pop should be sorted by fitness from low to high
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
        survivors = sorted_pop[round(self.survivor_rate*self.total_neurons):]
        parents = sorted_pop[round(self.parent_rate*self.total_neurons):]

        offspring = list(survivors)
        while len(offspring) < self.total_neurons:
            p0 = self.tournament_selection(parents)
            p1 = self.tournament_selection(parents)
            child = self.crossover(p0, p1)
            offspring.append(child)

        self.pop = np.array(offspring)
        self.mutate_all()

    # Log fitness stats
    def log(self, gen, network_fitnesses):
        fmin = network_fitnesses.min()
        fmean = network_fitnesses.mean()
        fmax = network_fitnesses.max()
        print(f"Generation {gen + 1} done. min: {fmin:.2f}, max: {fmax:.2f}, avg: {fmean:.2f}")
        with open(self.logpath, 'a') as logfile:
            logfile.write(f"{fmean}, {fmax}\n")

    # Run the GA
    def sane_execute(self, n_gens):
        for gen in range(n_gens):
            neuron_fitnesses, network_fitnesses = self.evaluate()
            self.log(gen, network_fitnesses)
            self.new_gen(neuron_fitnesses)
