################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman_sneaky')
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
    def __init__(self, n_hidden_neurons, enemy, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run, cores='max', current_generation = 0):

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
        self.total_sigma_data = []
        self.best = []
        self.run = run
        self.max_gain = -100*len(enemy)
        self.n_sigmas = 4
        self.cores = cores
        self.survival_number = 4
        self.current_generation = current_generation
        
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
                env.randomini = i
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
        if fitness_smop > avg_gains:
            surviving_player = True
        
        return [player, surviving_player, np.sum(gains), gains, fitness_smop, times, health_player, health_enemies, kills]
    
    def simulate(self, pop = []):
        '''
        Core function of the evo_algorithm dividing the games of a generation over the cores and storing/printing 
        the most important information of a generation is saved.
        '''
        #initiate 100 parents, the size of an agent is n_vars + sigmas

        if not len(pop) == self.population_size:
            DNA = np.random.uniform(-1, 1, (self.population_size ,self.n_vars))
            #set bias of shoot to 1
            for k in range(len(DNA)):
                DNA[k,213] = 1
            sigmas = np.random.uniform(0, 0.3, (self.population_size ,self.n_sigmas))
            pop = np.hstack((DNA, sigmas))
        
        avg_gains = self.max_gain
        
        for g in range(self.generations):
            gen_start = time.time()
            gains_array = []
            fitness_array = []
            surviving_players = []
            times_array = []
            health_array = []
            health_e_array = []
            kills_array = [] 
            sigma_array = []
            enemy = self.enemy

            
            #multiple cores implementation
            if self.cores == 'max':
                pool = mp.Pool(mp.cpu_count())
            else:
                pool = mp.Pool(cores)
                
            results = [pool.apply_async(self.play_game, args=(player, self.current_generation, avg_gains, enemy)) for player in pop]
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
                sigma_array.append(r[0][265:])
                
                #sigma data + some others
                self.total_sigma_data.append([self.current_generation]+list(np.concatenate([gains, kills, r[0][265:]]).flat))
                
                if avg_gain > self.max_gain:
                    self.max_gain = fitness
                    self.best = r[0]
                    
                if survive:
                    surviving_players.append(ind)
            pool.close()
            avg_gains = 0.25*np.mean(fitness_array) + 0.75*np.max(fitness_array)
#            if np.max(gains_array) > avg_gains:
#                if np.max(gains_array) < 0:
#                    avg_gains = 1.1*np.max(gains_array)
#                else:
#                    avg_gains = 0.9*np.max(gains_array)
#            else:
#                if avg_gains < 0:
#                    avg_gains *= 1.1
#                else:
#                    avg_gains *= 0.9
            
            self.total_data.append([np.max(gains_array), np.mean(gains_array), np.std(gains_array), np.max(fitness_array), np.mean(fitness_array), np.std(fitness_array)])
            
            
            ##survival of X best players
            best_players = np.sort(fitness_array, axis=None)[len(fitness_array) - self.survival_number]
            indexes = np.where(fitness_array >= best_players)[0]
            
            for index in indexes:
                if not index in surviving_players:
                    surviving_players.append(index)
            
            self.current_generation += 1
            #backup population each X gen
            if self.current_generation%10==9:
                self.backup_pop(pop, self.current_generation)
            
            
            pop = get_children(pop, surviving_players, np.array(fitness_array), self.mutation_baseline, self.mutation_multiplier)
            
            
            mean_sigmas = np.around(np.mean(np.array(pop)[:,265:], axis=0), decimals=2)
            max_sigmas = np.around(np.max(np.array(pop)[:,265:], axis=0), decimals=2)
            min_sigmas = np.around(np.min(np.array(pop)[:,265:], axis=0), decimals=2)
            print(f'Run: {self.run}, G: {self.current_generation}, F_mean = {round(np.mean(fitness_array),1)} pm {round(np.std(fitness_array),1)}, F_best = {round(np.max(fitness_array),1)}, G_mean = {np.round(np.mean(gains_array),1)}, G_best = {np.round(np.max(gains_array))}, S_mean={mean_sigmas} max:{max_sigmas} min:{min_sigmas}, kills={np.round(np.max(np.sum(kills_array, axis=1)),1)}, surviving={len(surviving_players)}, thr={np.round(avg_gains, 1)} time={round(time.time()-gen_start)}')
        return
    
    def save_results(self, full = False, append = False):
        writing_style = 'w'
        if append:
            writing_style = 'a'
        with open(f'data_normal/{self.experiment_name}/data_{self.run}.csv', writing_style, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.enemy, self.generations])
            writer.writerows(self.total_data)
        
        with open(f'data_normal/{self.experiment_name}/best_sol_{self.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.best)
            
        if full:
            title = ['generation']
            for enemy in self.enemy:
                title.append(f'gain_enemy_{enemy}')
            for enemy in self.enemy:
                title.append(f'kill_enemy_{enemy}')
            for sig in range(self.n_sigmas):
                title.append(f'sigma_{sig}')
            with open(f'data_normal/{self.experiment_name}/full_data_{self.run}.csv', writing_style, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not append:
                    writer.writerow(title)
                writer.writerows(self.total_sigma_data)
        return
    
    def backup_pop(self, population, generation):
        with open(f'data_normal/{self.experiment_name}/pop_backup_{generation}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(population)

def run_experiment(n_hidden_neurons, enemies, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, cores, new):
    start = time.time()
    for run in range(run_nr):
        evo = evo_algorithm(n_hidden_neurons, enemies, run_nr, generations, population_size, mutation_baseline, mutation_multiplier, repeats, fitter, run, cores)

        if new:
            #start a new run
            evo.simulate()
            evo.save_results(full=True)

        else:
            #continue an old run
            population = []
            load_from_generation = 0
            backup_name = f'data_normal/enemy_[1, 4, 6]_{fitter}/pop_backup_{load_from_generation}.csv'
            with open(backup_name, newline='', encoding='utf-8') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    population.append(row)
            evo.current_generation = load_from_generation
            evo.simulate(np.array(population))

            evo.save_results(full=True, append=False)


if __name__ == '__main__':
    n_hidden_neurons = 10       #number of hidden neurons
    enemies = [2, 6, 7, 8]               #which enemies
    run_nr = 1                  #number of runs
    generations = 300           #number of generations per run
    population_size = 100        #pop size
    mutation_baseline = 0       #minimal chance for a mutation event
    mutation_multiplier = 0.40  #fitness dependent multiplier of mutation chance
    repeats = 4
    fitter = 'standard'

    cores = 'max'
    new = True

