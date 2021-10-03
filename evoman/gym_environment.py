import math

import gym
import numpy as np
import pygame
from gym import spaces
from pygame.locals import *

import tmx as tmx
from gym_player import Player
from sensors import get_sensors


def action(num):
    actionset = []
    for i in range(5):
        actionset.append(num % 2)
        num = math.floor(num / 2)
    return actionset


class Evoman(gym.Env):
    HEIGHT = 512
    WIDTH = 736

    def load_sprites(self):

        # loads enemy and map
        self.enemy_import = __import__('gym_enemy' + str(self.enemyn))
        # self.enemy_import = imp.gym_enemy1
        self.tilemap = tmx.load(self.enemy_import.tilemap, self.screen.get_size())  # map

        self.sprite_e = tmx.SpriteLayer()
        start_cell = self.tilemap.layers['triggers'].find('enemy')[0]
        self.enemy = self.enemy_import.Enemy((start_cell.px, start_cell.py), self.sprite_e)
        self.tilemap.layers.append(self.sprite_e)  # enemy

        # loads player
        self.sprite_p = tmx.SpriteLayer()
        start_cell = self.tilemap.layers['triggers'].find('player')[0]
        self.player = Player((start_cell.px, start_cell.py), self.sprite_p)
        self.tilemap.layers.append(self.sprite_p)

    def __init__(self,
                 enemyn=1,
                 level=2,
                 enemymode="static",
                 contacthurt="player",
                 randomini=False,
                 logs=False,
                 savelogs=False,
                 precise_clock=False,
                 timeexpire=3000,
                 overturetime=100,
                 cost_per_timestep=0.0,
                 ):
        super(Evoman, self).__init__()
        self.action_space = spaces.MultiBinary(5)
        observation_space_low = np.array([-self.HEIGHT if a % 2 else -self.WIDTH for a in range(20)], dtype=np.float32)
        observation_space_low[2] = -2
        observation_space_low[3] = -2
        observation_space_high = np.array([self.HEIGHT if a % 2 else self.WIDTH for a in range(20)], dtype=np.float32)
        observation_space_high[2] = 1
        observation_space_high[3] = 1

        self.observation_space = spaces.Box(
            low=observation_space_low,
            high=observation_space_high,
            dtype=np.float32
        )
        self.enemyn = enemyn
        self.level = level
        self.enemymode = enemymode
        self.contacthurt = contacthurt
        self.randomini = randomini
        self.logs = logs
        self.savelogs = savelogs
        self.precise_clock = precise_clock
        self.timeexpire = timeexpire
        self.overturetime = overturetime
        self.cost_per_timestep = cost_per_timestep

        # compatibility with existing Player and Enemy classes
        self.playermode = 'ai'

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), DOUBLEBUF)
        self.screen.set_alpha(None)
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP])
        self.reset()

    def reset(self):
        self.time = 0
        self.freeze_p = False
        self.freeze_e = False
        self.load_sprites()
        return get_sensors(self)

    def step(self, actions):
        self.reward = -self.cost_per_timestep
        self.actions = actions
        done = False

        self.time += 1
        self.tilemap.update(33 / 1000., self)

        if self.player.life == 0 or self.enemy.life == 0:
            done = True

            self.enemy.kill()
            self.player.kill()

        if self.time >= self.timeexpire:
            done = True

        info = {}

        return get_sensors(self), self.reward, done, info

    def render(self, mode="human"):
        # updates objects and draws its itens on screen
        self.screen.fill((250, 250, 250))
        self.tilemap.draw(self.screen)

        # player life bar
        vbar = int(100 * (1 - (self.player.life / float(self.player.max_life))))
        pygame.draw.line(self.screen, (0, 0, 0), [40, 40], [140, 40], 2)
        pygame.draw.line(self.screen, (0, 0, 0), [40, 45], [140, 45], 5)
        pygame.draw.line(self.screen, (150, 24, 25), [40, 45], [140 - vbar, 45], 5)
        pygame.draw.line(self.screen, (0, 0, 0), [40, 49], [140, 49], 2)

        # enemy life bar
        vbar = int(100 * (1 - (self.enemy.life / float(self.enemy.max_life))))
        pygame.draw.line(self.screen, (0, 0, 0), [590, 40], [695, 40], 2)
        pygame.draw.line(self.screen, (0, 0, 0), [590, 45], [695, 45], 5)
        pygame.draw.line(self.screen, (194, 118, 55), [590, 45], [695 - vbar, 45], 5)
        pygame.draw.line(self.screen, (0, 0, 0), [590, 49], [695, 49], 2)

        if mode == "bgr":
            pygame.display.flip()
            pxarray = pygame.PixelArray(pygame.display.get_surface())
            frame = np.array(pxarray)
            rows = lambda val: np.array([val % 256, (val // 256) % 256, (val // (256 * 256)) % 256], dtype=np.uint8)
            cols = lambda val: rows(val)

            frame = np.transpose(cols(frame), (2, 1, 0))

            return frame

        if mode == "human":
            if self.precise_clock:
                self.clock.tick_busy_loop(30)
            else:
                self.clock.tick(30)
            pygame.display.flip()
            return

        raise NotImplementedError()
