from statistics import mean 

class generation_reporter():
    
    def __init__(self, logpath):
        self.logpath = logpath
        with open(self.logpath, "a") as data:
            data.truncate()

    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        # collect a list of all fitness values in the population
        fitness_list = [individual.fitness for individual in population.values()]
        fitness_list = [i for i in fitness_list if i is not None]
        
        # calculate mean and max fitness
        fitness_mean = str(mean(fitness_list))
        fitness_max = str(max(fitness_list))
        
        # write new entry to the data file
        data = open(self.logpath, "a")
        data.write(fitness_mean + " " + fitness_max + "\n")
        data.close()

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass