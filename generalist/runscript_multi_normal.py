################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from bio_functions import crossover, mutation, get_children, fitfunc
import csv
import time
import multiprocessing as mp

# choose this for not using visuals and thus making experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

#make sure to not print every startup of the pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"    


class evo_algorithm:
    '''
    The main Class containing the algorithm for a set of inputs.
    
    
    enemy               = [list] list of enemies (can be a single one) to train on
    run_nr              = [int] total number of runs, a run is a complete iteration of X generations
    generations         = [int] number of generations per run
    population_size     = [int] size of the population
    mutation_baseline   = [float] minimal chance for a mutation
    mutation_multiplier = [float] fitness dependent multiplier of mutation chance
    repeats             = [int] number of repeated fight of an agent within a generation
    fitter              = [string] name of the fitter to use
    experiment_name     = [string] the used name to save the data to
    n_hidden_neurons    = [int] number of hidden neurons in the hidden layer
    n_vars              = [int] total number of variables needed for the neural network
    total_data          = [list] place to save data to for later printing to csv\
    best                = [array] the best solution
    n_sigmas            = [int] number of sigmas to use (only use 1 or 4)
    '''
    def __init__(self, n_hidden_neurons, enemy, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run):
        self.enemy = enemy          #which enemy
        self.run_nr = run_nr                  #number of runs
        self.generations = generations           #number of generations per run
        self.population_size = population_size       #pop size
        self.mutation_baseline = mutation_baseline    #minimal chance for a mutation event
        self.mutation_multiplier = mutation_multiplier  #fitness dependent multiplier of mutation chance
        self.repeats = repeats
        self.fitter = fitter
        self.experiment_name = f'enemy_{enemy}_{fitter}'
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (20+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        self.total_data = []
        self.best = []
        self.run = run
        self.max_gain = -100*len(enemy)
        self.n_sigmas = 4
        
        #make save folder
        if not os.path.exists(f'data_normal/{self.experiment_name}'):
            os.makedirs(f'data_normal/{self.experiment_name}')  
    
    def play_game(self, player, g, avg_gains, enemies):
        '''
        A function that starts a game against a given set of enemies for a number of repeats in order to evaluate a specific solution vector.
        
        Inputs:
            player      = [array] the DNA (weights + biases neural network) and the sigmas
            g           = [int]   the current generation
            avg_fitness = [float] the avg_fitness of the previous generation
            enemies     = [list]  the enemies to train on
            
        Outputs:
            player      = [array] the DNA
            surviving_player = [boolean] if the player survived or not
            mean_gains  = [float] average gains
            gains       = [list]  the gains of the individual enemies
            fitness_smop= [float] average fitness values
            times       = [list]  the runtimes
            healths     = [list]  healths_player
            healths_e   = [list]  healths_enemies
            kills       = [list]  list with kill fraction per enemy
        '''
        
        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=f'data_normal/{self.experiment_name}',
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini = "yes")
        fitness_smop = 0
        repeats = self.repeats
        gains = []
        health_player = []
        health_enemies = []
        times = []
        kills = []
        surviving_player = False
        
        
        #battle each enemy in this list
        for enemy in enemies:
            gain_avg = 0
            time_avg = 0
            health = 0
            health_enemy = 0
            kill = 0
            
            #repeat each player to counter the randomness
            for i in range(repeats):
                f, p, e, t = env.run_single(enemy, pcont=player[:self.n_vars], econt="None")
                
                fitness_smop += fitfunc(self.fitter, self.generations, g, t, e, p)*(1/repeats)
                health += (1/repeats)*p
                health_enemy += (1/repeats)*e
                time_avg += (1/repeats)*t
                gain_avg += (1/repeats)*p - (1/repeats)*e
                if e == 0:
                    kill += (1/repeats)
                
            gains.append(gain_avg)
            health_player.append(health)
            health_enemies.append(health_enemy)
            times.append(time_avg)
            kills.append(kill)
            
        #if nog good enough 'die'
        if np.mean(gains) > avg_gains:
            surviving_player = True
        
        return [player, surviving_player, np.mean(gains), gains, fitness_smop, times, health_player, health_enemies, kills]
    
    def simulate(self, pop = []):
        #initiate 100 parents, the size of an agent is n_vars + sigmas
        if not len(pop) == self.population_size:
            DNA = np.random.uniform(-1, 1, (self.population_size ,self.n_vars))
            sigmas = np.random.uniform(0, 1, (self.population_size ,self.n_sigmas))
            pop = np.hstack((DNA, sigmas))
        avg_gains = 0
        
        for g in range(self.generations):
            gen_start = time.time()
            gains_array = []
            fitness_array = []
            surviving_players = []
            times_array = []
            health_array = []
            health_e_array = []
            kills_array = [] 
            enemy = self.enemy
            
            #multiple cores implementation
            pool = mp.Pool(mp.cpu_count())
            results = [pool.apply_async(self.play_game, args=(player, g, avg_gains, enemy)) for player in pop]
            pop = []
            
            #retrieve all the data
            for ind, result in enumerate(results):
                r = result.get()
                pop.append(r[0])
                survive = r[1]
                avg_gain = r[2]
                gains = r[3]
                fitness = r[4]
                times = r[5]
                health = r[6]
                health_e = r[7]
                kills = r[8]
                
                #save all the data to arrays
                gains_array.append(avg_gain)
                fitness_array.append(fitness)
                times_array.append(times)
                health_array.append(health)
                health_e_array.append(health_e)
                kills_array.append(kills)
                
                if avg_gain > self.max_gain:
                    self.max_gain = avg_gain
                    self.best = r[0]
                    
                if survive:
                    surviving_players.append(ind)
            pool.close()
            
            if avg_gain > avg_gains:
                avg_gains = avg_gain
            else:
                avg_gains *= 0.9
            
            self.total_data.append([np.max(gains_array), np.mean(gains_array), np.std(gains_array), np.max(fitness_array), np.mean(fitness_array), np.std(fitness_array)])
            
            
            pop = get_children(pop, surviving_players, np.array(fitness_array),
                               mutation_baseline, mutation_multiplier)
            
            mean_sigmas = np.around(np.mean(np.array(pop)[:,265:], axis=0), decimals=2)
            max_sigmas = np.around(np.max(np.array(pop)[:,265:], axis=0), decimals=2)
            min_sigmas = np.around(np.min(np.array(pop)[:,265:], axis=0), decimals=2)
            print(f'Run: {self.run}, Generation {g}, fitness_mean = {round(np.mean(fitness_array),2)} pm {round(np.std(fitness_array),2)}, fitness_best = {round(np.max(fitness_array),2)}, mean_gain = {np.round(np.mean(gains_array),2)}, best_gain_tot = {np.round(self.max_gain,2)}, mean_s={mean_sigmas} max:{max_sigmas} min:{min_sigmas}, time={round(time.time()-gen_start)}')
        return
    
    def save_results(self, extended = False, full = False):
        with open(f'data_normal/{self.experiment_name}/data_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.enemy, self.generations])
            writer.writerows(self.total_data)
        
        with open(f'data_normal/{self.experiment_name}/best_sol_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.best)          
        return


if __name__ == '__main__':
    n_hidden_neurons = 10       #number of hidden neurons
    enemies = [1,2,5]           #which enemies
    run_nr = 1                  #number of runs
    generations = 100           #number of generations per run
    population_size = 100       #pop size
    mutation_baseline = 0.02    #minimal chance for a mutation event
    mutation_multiplier = 0.20  #fitness dependent multiplier of mutation chance
    repeats = 4
    fitter = 'standard'
    start = time.time()
    
    for run in range(run_nr):
        evo = evo_algorithm(n_hidden_neurons, enemies, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run)
        evo.simulate()
        evo.save_results()