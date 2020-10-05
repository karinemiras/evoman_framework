import neat
import numpy as np
import sys, os, glob
import random
sys.path.insert(0, 'evoman') 
from environment import Environment
from neat_controller import player_controller
import time
import pickle
from math import fabs,sqrt
import multiprocessing
import csv

class generalist:
    def __init__(self,config='.',maxgen=50,popsize=50,fitness_cutoff=120,EA='neat',group=[2,4],subname='exp1'):

        self.maxgen = maxgen # cutoff number of generations
        self.fitness_cutoff = fitness_cutoff
        self.pop_size = popsize
        self.EA = EA
        self.group = group
        self.subname = subname
        self.config = config

        #LOAD AND  CHANGE CONFIGS
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,self.config)
        self.config.pop_size = self.pop_size
        self.config.fitness_threshold = self.fitness_cutoff



        #creating a folder for results
        self.experiment_name = 'EA' + self.EA + '/engroup%i%i'%(self.group[0],self.group[1])
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)
            
        # initializes simulation in multi evolution mode, for multiple static enemies.
        self.env = Environment(experiment_name=self.experiment_name,enemies=self.group, multiplemode="yes",playermode="ai"
                               ,enemymode="static",player_controller=player_controller(),level=2,speed="fastest")

        def new_fitness(self,values):
            '''The original cons_multi func in environment.py calculates
            fitness=mean(enemies)-std(enemies). Idk why it subtracts std, so i modify it here.
            Unsure if it's allowed...
            '''
            return values.mean()

        self.env.cons_multi = new_fitness.__get__(self.env)

    def eval_genome(self,genome, config):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness,player,energy,time = self.env.play(pcont=net)

        return fitness

        
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)
    
    def my_save_func(self,delimiter=' ',filename='fitness_history.csv'):
        """ Write our own function that saves the 
        population's best, average and std of fitness per gen"""
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)

            best_fitness = [c.fitness for c in self.stats.most_fit_genomes]
            avg_fitness = self.stats.get_fitness_mean()
            std_fitness = self.stats.get_fitness_stdev()
            for best, avg, std in zip(best_fitness, avg_fitness,std_fitness):
                w.writerow([best, avg, std])

    def run(self):
        #creating population
        self.p = neat.Population(self.config)
        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(True))

        # Add reporter to collect stats
        self.stats = neat.StatisticsReporter()
        self.p.add_reporter(self.stats)
        
        #self.winner = self.p.run(self.eval_genomes,self.maxgen)

        #run NEAT
        self.p.run(self.eval_genomes,self.maxgen)

        # Write run statistics to CSV file
        self.my_save_func(delimiter=',', filename=self.experiment_name+'/'+ self.subname+ '_output.csv')

        # Display the winning genome
        self.winner = self.stats.best_genome()
        print('\nBest genome:\nfitness {!s}\n{!s}'.format(self.winner.fitness, self.winner))

        # Save winning genomes topology and weights to load for testing
        pickle.dump(self.winner, open(self.experiment_name+'/'+ self.subname+"_winner.p","wb"))

experiments = [1,2,3,4,5,6,7,8,9,10]
ea = 'neat' #ENTER WHICH EA YOU ARE RUNNING ('neat' or 'cov')
groups = [[2,4],[2,6]] 
group = groups[1]#ENTER WHICH group YOU ARE RUNNING (1 or 2)

#RUN EXPERIMENT
if __name__ == '__main__':
    local_dir = os.path.dirname(os.path.abspath("__file__"))
    config_path = os.path.join(local_dir, 'config_file.ini')
    

    #here we can implement loop to run experiments 1 to 10 
    # random.seed(experiments[3])
    # test = generalist(config=config_path,maxgen=2,popsize=3,EA=ea,group=group,subname='/exp%i'%experiments[0])
    # test.run()

if __name__ == '__main__':
    local_dir = os.path.dirname(os.path.abspath("__file__"))
    config_path = os.path.join(local_dir, 'config_file.ini')
    

    #here we can implement loop to run experiments 1 to 10 
    for i in range(0,len(experiments)):
        random.seed(experiments[i])
        test = generalist(config=config_path,maxgen=50,popsize=50,EA=ea,group=group,subname='/exp%i'%experiments[i])
        test.run()
    