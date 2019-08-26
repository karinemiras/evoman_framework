################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys
import gzip
import pickle
import numpy
import pygame
from pygame.locals import *
import struct
import tmx

from player import *
from controller import Controller


# main class
class Environment(object):


    # simulation parameters
    def __init__(self,
                 experiment_name='test',
                 multiplemode="no",           # yes or no
                 enemies=[1],                 # array with 1 to 8 items, values from 1 to 8
                 loadplayer="yes",            # yes or no
                 loadenemy="yes",             # yes or no
                 level=2,                     # integer
                 playermode="ai",             # ai or human
                 enemymode="static",          # ai or static
                 speed="fastest",             # normal or fastest
                 inputscoded="no",            # yes or no
                 randomini="no",              # yes or no
                 sound="on",                  # on or off
                 contacthurt="player",        # player or enemy
                 logs="on",                   # on or off
                 savelogs="yes",              # yes or no
                 clockprec="low",
                 timeexpire=3000,             # integer
                 overturetime=100,            # integer
                 solutions=None,              # any
                 player_controller=None,      # controller object
                 enemy_controller=None      ):# controller object


        # initializes parameters

        self.experiment_name = experiment_name
        self.multiplemode = multiplemode
        self.enemies = enemies
        self.enemyn = enemies[0] # initial current enemy
        self.loadplayer = loadplayer
        self.loadenemy = loadenemy
        self.level = level
        self.playermode = playermode
        self.enemymode = enemymode
        self.speed = speed
        self.inputscoded = inputscoded
        self.randomini = randomini
        self.sound = sound
        self.contacthurt = contacthurt
        self.logs = logs
        self.savelogs = savelogs
        self.clockprec = clockprec
        self.timeexpire = timeexpire
        self.overturetime = overturetime
        self.solutions = solutions


        # initializes default random controllers

        if self.playermode == "ai" and player_controller == None:
            self.player_controller = Controller()
        else:
            self.player_controller =  player_controller

        if self.enemymode == "ai" and enemy_controller == None:
            self.enemy_controller = Controller()
        else:
            self.enemy_controller =  enemy_controller


        # initializes log file
        if self.logs  == "on" and self.savelogs == "yes":
            file_aux  = open(self.experiment_name+'/evoman_logs.txt','w')
            file_aux.close()


        # initializes pygame library
        pygame.init()
        self.print_logs("MESSAGE: Pygame initialized for simulation.")

        # initializes sound library for playing mode
        if self.sound == "on" and self.playermode == "human":
            pygame.mixer.init()
            self.print_logs("MESSAGE: sound has been turned on.")

        # initializes joystick library
        if self.playermode == "human":
            pygame.joystick.init()
            self.joy =  pygame.joystick.get_count()

        self.clock = pygame.time.Clock() # initializes game clock resource


        # generates screen
        if self.playermode == 'human': # playing mode in fullscreen
            flags =  DOUBLEBUF  |  FULLSCREEN
        else:
            flags =  DOUBLEBUF

        self.screen = pygame.display.set_mode((736, 512), flags)

        self.screen.set_alpha(None) # disables uneeded alpha
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP]) # enables only needed events

        self.load_sprites()



    def load_sprites(self):

        # loads enemy and map
        enemy = __import__('enemy'+str(self.enemyn))
        self.tilemap = tmx.load(enemy.tilemap, self.screen.get_size())  # map

        self.sprite_e = tmx.SpriteLayer()
        start_cell = self.tilemap.layers['triggers'].find('enemy')[0]
        self.enemy = enemy.Enemy((start_cell.px, start_cell.py), self.sprite_e)
        self.tilemap.layers.append(self.sprite_e)  # enemy

        # loads player
        self.sprite_p = tmx.SpriteLayer()
        start_cell = self.tilemap.layers['triggers'].find('player')[0]
        self.player = Player((start_cell.px, start_cell.py), self.enemyn, self.level, self.sprite_p)
        self.tilemap.layers.append(self.sprite_p)

        self.player.sensors = Sensors()
        self.enemy.sensors = Sensors()


    # updates environment with backup of current solutions in simulation
    def get_solutions(self):
        return self.solutions


        # method for updating solutions bkp in simulation
    def update_solutions(self, solutions):
        self.solutions = solutions


    # method for updating simulation parameters
    def update_parameter(self, name, value):

        if type(value) is str:
            exec('self.'+name +"= '"+ value+"'")
        else:
            exec('self.'+name +"= "+ str(value))

        self.print_logs("PARAMETER CHANGE: "+name+" = "+str(value))



    def print_logs(self, msg):
        if self.logs == "on":
            print('\n'+msg) # prints log messages to screen

            if self.savelogs == "yes": # prints log messages to file
                file_aux  = open(self.experiment_name+'/evoman_logs.txt','a')
                file_aux.write('\n\n'+msg)
                file_aux.close()


    def get_num_sensors(self):

        if hasattr(self, 'enemy') and self.enemymode == "ai":
            return  len(self.enemy.sensors.get(self))
        else:
            if hasattr(self, 'player') and self.playermode == "ai":
                return len(self.player.sensors.get(self))
            else:
                return 0


    # writes all variables related to game state into log
    def state_to_log(self):


        self.print_logs("########## Simulation state - INI ###########")
        if self.solutions == None:
            self.print_logs("# solutions # : EMPTY ")
        else:
            self.print_logs("# solutions # : LOADED ")

        self.print_logs("# sensors # : "+ str( self.get_num_sensors() ))
        self.print_logs(" ------  parameters ------  ")
        self.print_logs("# contact hurt (training agent) # : "  +self.contacthurt)

        self.print_logs("multiple mode: "+self.multiplemode)

        en = ''
        for e in self.enemies:
            en += ' '+str(e)
        self.print_logs("enemies list:"+ en)

        self.print_logs("current enemy: " +str(self.enemyn))
        self.print_logs("player mode: " +self.playermode)
        self.print_logs("enemy mode: "  +self.enemymode)
        self.print_logs("level: " +str(self.level))
        self.print_logs("clock precision: "+ self.clockprec)
        self.print_logs("inputs coded: "  +self.inputscoded)
        self.print_logs("random initialization: "  +self.randomini)
        self.print_logs("expiration time: "  +str(self.timeexpire))
        self.print_logs("speed: " +self.speed)
        self.print_logs("load player: " +self.loadplayer)
        self.print_logs("load enemy: " +self.loadenemy)
        self.print_logs("sound: "  +self.sound)
        self.print_logs("overture time: "  +str(self.overturetime))
        self.print_logs("logs: "+self.logs)
        self.print_logs("save logs: "+self.savelogs)
        self.print_logs("########## Simulation state - END ###########")



    # exports current environment state to files
    def save_state(self):

        # saves configuration file for simulation parameters
        file_aux  = open(self.experiment_name+'/evoman_paramstate.txt','w')
        en = ''
        for e in self.enemies:
            en += ' '+str(e)
        file_aux.write("\nenemies"+ en)
        file_aux.write("\ntimeexpire "  +str(self.timeexpire))
        file_aux.write("\nlevel " +str(self.level))
        file_aux.write("\nenemyn " +str(self.enemyn))
        file_aux.write("\noverturetime "  +str(self.overturetime))
        file_aux.write("\nplayermode " +self.playermode)
        file_aux.write("\nenemymode "  +self.enemymode)
        file_aux.write("\ncontacthurt "  +self.contacthurt)
        file_aux.write("\nclockprec "+ self.clockprec)
        file_aux.write("\ninputscoded "  +self.inputscoded)
        file_aux.write("\nrandomini "  +self.randomini)
        file_aux.write("\nmultiplemode "+self.multiplemode)
        file_aux.write("\nspeed " +self.speed)
        file_aux.write("\nloadplayer " +self.loadplayer)
        file_aux.write("\nloadenemy " +self.loadenemy)
        file_aux.write("\nsound "  +self.sound)
        file_aux.write("\nlogs "+self.logs)
        file_aux.write("\nsavelogs "+self.savelogs)
        file_aux.close()

        # saves state of solutions in the simulation
        file = gzip.open(self.experiment_name+'/evoman_solstate', 'w', compresslevel = 5)
        pickle.dump(self.solutions, file, protocol=2)
        file.close()


        self.print_logs("MESSAGE: state has been saved to files.")



    # loads a state for environment from files
    def load_state(self):


        try:

            # loads parameters
            state = open(self.experiment_name+'/evoman_paramstate.txt','r')
            state = state.readlines()
            for idp,p in enumerate(state):
                pv = p.split(' ')

                if idp>0:    # ignore first line
                    if idp==1: # enemy list
                        en = []
                        for i in range(1,len(pv)):
                            en.append(int(pv[i].rstrip('\n')))
                        self.update_parameter(pv[0], en)
                    elif idp<6: # numeric params
                        self.update_parameter(pv[0], int(pv[1].rstrip('\n')))
                    else: # string params
                        self.update_parameter(pv[0], pv[1].rstrip('\n'))

            # loads solutions
            file = gzip.open(self.experiment_name+'/evoman_solstate')
            self.solutions =  pickle.load(file, encoding='latin1')
            self.print_logs("MESSAGE: state has been loaded.")

        except IOError:
            self.print_logs("ERROR: could not load state.")




    def checks_params(self):

        # validates parameters values

        if self.multiplemode == "yes" and len(self.enemies) < 2:
            self.print_logs("ERROR: 'enemies' must contain more than one enemy for multiple mode.")
            sys.exit(0)

        if self.enemymode not in ('static','ai'):
            self.print_logs("ERROR: 'enemy mode' must be 'static' or 'ai'.")
            sys.exit(0)

        if self.playermode not in ('human','ai'):
            self.print_logs("ERROR: 'player mode' must be 'human' or 'ai'.")
            sys.exit(0)

        if self.loadplayer not in ('yes','no'):
            self.print_logs("ERROR: 'load player' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.loadenemy not in ('yes','no'):
            self.print_logs("ERROR: 'load enemy' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.inputscoded not in ('yes','no'):
            self.print_logs("ERROR: 'inputs coded' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.multiplemode not in ('yes','no'):
            self.print_logs("ERROR: 'multiplemode' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.randomini not in ('yes','no'):
            self.print_logs("ERROR: 'random ini' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.savelogs not in ('yes','no'):
            self.print_logs("ERROR: 'save logs' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.speed not in ('normal','fastest'):
            self.print_logs("ERROR: 'speed' value must be 'normal' or 'fastest'.")
            sys.exit(0)

        if self.logs not in ('on','off'):
            self.print_logs("ERROR: 'logs' value must be 'on' or 'off'.")
            sys.exit(0)

        if self.clockprec not in ('low','medium'):
            self.print_logs("ERROR: 'clockprec' value must be 'low' or 'medium'.")
            sys.exit(0)

        if self.sound not in ('on','off'):
            self.print_logs("ERROR: 'sound' value must be 'on' or 'off'.")
            sys.exit(0)

        if self.contacthurt not in ('player','enemy'):
            self.print_logs("ERROR: 'contacthurt' value must be 'player' or 'enemy'.")
            sys.exit(0)

        if type(self.timeexpire) is not int:
            self.print_logs("ERROR: 'timeexpire' must be integer.")
            sys.exit(0)

        if type(self.level) is not int:
            self.print_logs("ERROR: 'level' must be integer.")
            sys.exit(0)

        if type(self.overturetime) is not int:
            self.print_logs("ERROR: 'overturetime' must be integer.")
            sys.exit(0)


        # checks parameters consistency

        if self.multiplemode == "no" and len(self.enemies) > 1:
            self.print_logs("MESSAGE: there is more than one enemy in 'enemies' list although the mode is not multiple.")

        if self.level < 1 or self.level > 3:
            self.print_logs("MESSAGE: 'level' chosen is out of recommended (tested).")




            # default fitness function for single solutions
    def fitness_single(self):
        return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - numpy.log(self.get_time())

    # default fitness function for consolidating solutions among multiple games
    def cons_multi(self,values):
        return values.mean() - values.std()

    # measures the energy of the player
    def get_playerlife(self):
        return self.player.life

    # measures the energy of the enemy
    def get_enemylife(self):
        return self.enemy.life

    # gets run time
    def get_time(self):
        return self.time


    # runs game for a single enemy
    def run_single(self,enemyn,pcont,econt):

        # sets controllers
        self.pcont = pcont
        self.econt = econt

        self.checks_params()


        self.enemyn = enemyn # sets the current enemy
        ends = 0
        self.time = 0
        self.freeze_p = False
        self.freeze_e = False
        self.start = False

        enemy = __import__('enemy'+str(self.enemyn))

        self.load_sprites()


        # game main loop

        while 1:

            # adjusts frames rate for defining game speed

            if self.clockprec == "medium":  # medium clock precision
                if self.speed == 'normal':
                    self.clock.tick_busy_loop(30)
                elif self.speed == 'fastest':
                    self.clock.tick_busy_loop()

            else:   # low clock precision

                if self.speed == 'normal':
                    self.clock.tick(30)
                elif self.speed == 'fastest':
                    self.clock.tick()


            # game timer
            self.time += 1
            if self.playermode == "human":

                # sound effects
                if self.sound == "on" and self.time == 1:
                    sound = pygame.mixer.Sound('evoman/sounds/open.wav')
                    c = pygame.mixer.Channel(1)
                    c.set_volume(1)
                    c.play(sound,loops=10)

                if self.time >  self.overturetime: # delays game start a little bit for human mode
                    self.start = True
            else:
                self.start = True


            # checks screen closing button
            self.event = pygame.event.get()
            for event in  self.event:
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            # updates objects and draws its itens on screen
            self.screen.fill((250,250,250))
            self.tilemap.update( 33 / 1000., self)
            self.tilemap.draw(self.screen)

            # player life bar
            vbar = int(100 *( 1-(self.player.life/float(self.player.max_life)) ))
            pygame.draw.line(self.screen, (0,   0,   0), [40, 40],[140, 40], 2)
            pygame.draw.line(self.screen, (0,   0,   0), [40, 45],[140, 45], 5)
            pygame.draw.line(self.screen, (150,24,25),   [40, 45],[140 - vbar, 45], 5)
            pygame.draw.line(self.screen, (0,   0,   0), [40, 49],[140, 49], 2)

            # enemy life bar
            vbar = int(100 *( 1-(self.enemy.life/float(self.enemy.max_life)) ))
            pygame.draw.line(self.screen, (0,   0,   0), [590, 40],[695, 40], 2)
            pygame.draw.line(self.screen, (0,   0,   0), [590, 45],[695, 45], 5)
            pygame.draw.line(self.screen, (194,118,55),  [590, 45],[695 - vbar, 45], 5)
            pygame.draw.line(self.screen, (0,   0,   0), [590, 49],[695, 49], 2)


            #gets fitness for training agents
            fitness = self.fitness_single()


            # returns results of the run
            def return_run():
                self.print_logs("RUN: run status: enemy: "+str(self.enemyn)+"; fitness: " + str(fitness) + "; player life: " + str(self.player.life)  + "; enemy life: " + str(self.enemy.life) + "; time: " + str(self.time))

                return  fitness, self.player.life, self.enemy.life, self.time



            if self.start == False and self.playermode == "human":

                myfont = pygame.font.SysFont("Comic sams", 100)
                pygame.font.Font.set_bold
                self.screen.blit(myfont.render("Player", 1,  (150,24,25)), (50, 180))
                self.screen.blit(myfont.render("  VS  ", 1,  (50,24,25)), (250, 180))
                self.screen.blit(myfont.render("Enemy "+str(self.enemyn), 1,  (194,118,55)), (400, 180))


            # checks player life status
            if self.player.life == 0:
                ends -= 1

                # tells user that player has lost
                if self.playermode == "human":
                    myfont = pygame.font.SysFont("Comic sams", 100)
                    pygame.font.Font.set_bold
                    self.screen.blit(myfont.render(" Enemy wins", 1, (194,118,55)), (150, 180))

                self.player.kill() # removes player sprite
                self.enemy.kill()  # removes enemy sprite

                if self.playermode == "human":
                    # delays run finalization for human mode
                    if ends == -self.overturetime:
                        return return_run()
                else:
                    return return_run()


            # checks enemy life status
            if self.enemy.life == 0:
                ends -= 1

                self.screen.fill((250,250,250))
                self.tilemap.draw(self.screen)

                # tells user that player has won
                if self.playermode == "human":
                    myfont = pygame.font.SysFont("Comic sams", 100)
                    pygame.font.Font.set_bold
                    self.screen.blit(myfont.render(" Player wins ", 1, (150,24,25) ), (170, 180))

                self.enemy.kill()   # removes enemy sprite
                self.player.kill()  # removes player sprite

                if self.playermode == "human":
                    if ends == -self.overturetime:
                        return return_run()
                else:
                    return return_run()


            if self.loadplayer == "no":# removes player sprite from game
                self.player.kill()

            if self.loadenemy == "no":  #removes enemy sprite from game
                self.enemy.kill()

                # updates screen
            pygame.display.flip()


            # game runtime limit
            if self.playermode == 'ai':
                if self.time >= enemy.timeexpire:
                    return return_run()

            else:
                if self.time >= self.timeexpire:
                    return return_run()



    # repeats run for every enemy in list
    def multiple(self,pcont,econt):

        vfitness, vplayerlife, venemylife, vtime = [],[],[],[]
        for e in self.enemies:

            fitness, playerlife, enemylife, time  = self.run_single(e,pcont,econt)
            vfitness.append(fitness)
            vplayerlife.append(playerlife)
            venemylife.append(enemylife)
            vtime.append(time)

        vfitness = self.cons_multi(numpy.array(vfitness))
        vplayerlife = self.cons_multi(numpy.array(vplayerlife))
        venemylife = self.cons_multi(numpy.array(venemylife))
        vtime = self.cons_multi(numpy.array(vtime))

        return    vfitness, vplayerlife, venemylife, vtime


    # checks objective mode
    def play(self,pcont="None",econt="None"):

        if self.multiplemode == "yes":
            return self.multiple(pcont,econt)
        else:
            return self.run_single(self.enemies[0],pcont,econt)
