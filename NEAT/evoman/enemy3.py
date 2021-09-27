################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys
import numpy
import random

import Base
from Base.SpriteConstants import *
from Base.SpriteDefinition import *
from sensors import Sensors

tilemap = 'evoman/map2.tmx'
timeexpire = 1000 # game run limit

# enemy 3 sprite, woodman
class Enemy(pygame.sprite.Sprite):


    def __init__(self, location,*groups):
        super(Enemy, self).__init__(*groups)
        self.spriteDefinition = SpriteDefinition('evoman/images/EnemySprites.png', 0, 0, 43, 59)
        self.updateSprite(SpriteConstants.STANDING, SpriteConstants.LEFT)

        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = -1
        self.max_life = 100
        self.life = self.max_life
        self.resting = 0
        self.dy = 0
        self.twists = []
        self.alternate = 1
        self.imune = 0
        self.timeenemy = 0
        self.hurt = 0
        self.shooting = 0
        self.gun_cooldown = 0


    def update(self, dt, game):


        if game.time==1:
            # puts enemy in random initial position
            if game.randomini == 'yes':
                self.rect.x = numpy.random.choice([640,500,400,300])

        # defines game mode for player action
        if game.enemymode == 'static': # enemy controlled by static movements

            if self.timeenemy >= 120  and self.timeenemy <= 140:
                atack1 = 1
            else:
                atack1 = 0

            if self.timeenemy == 130:
                atack2 = 1
            else:
                atack2 = 0

            if self.timeenemy> 140:
                atack3 = 1
            else:
                atack3 = 0

            if self.timeenemy == 30:
                atack4 = 1
            else:
                atack4 = 0


        elif game.enemymode == 'ai': # player controlled by AI algorithm


            # calls the controller providing game sensors
            actions = game.enemy_controller.control(self.sensors.get(game), game.econt)
            if len(actions) < 4:
                game.print_logs("ERROR: Enemy 1 controller must return 4 decision variables.")
                sys.exit(0)

            atack1 = actions[0]
            atack2 = actions[1]
            atack3 = actions[2]
            atack4 = actions[3]


            if atack4 == 1 and not self.gun_cooldown:
                atack4 = 1
            else:
                atack4 = 0


        # if 'start game' is true
        if game.start == 1:

            self.timeenemy += 1 # increments enemy timer


            # copies last position state of the enemy
            last = self.rect.copy()

            # movements of the enemy on the axis x
            if atack1 == 1:

                self.rect.x += self.direction * 180 * dt  # goes forward
                # jumps
                if atack2 == 1 and self.resting == 1:
                    self.dy = -700
                    self.resting = 0

                # animation, running enemy images alternatetion
                if self.direction > 0:
                    direction = SpriteConstants.RIGHT
                else:
                    direction = SpriteConstants.LEFT

                if self.alternate == 1:
                    self.updateSprite(SpriteConstants.START_RUNNING, direction)
                if self.alternate == 4 or self.alternate == 10:
                    self.updateSprite(SpriteConstants.RUNNING_STEP1, direction)
                if self.alternate == 7:
                    self.updateSprite(SpriteConstants.RUNNING_STEP2, direction)

                self.alternate += 1
                if self.alternate > 12:
                    self.alternate = 1

                #  changes the image when enemy jumps
                if self.resting == 0:
                   if self.direction == -1:
                       self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.LEFT)
                   else:
                       self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.RIGHT)

            else:
                # animation, standing up images
                if self.direction == -1:
                   self.updateSprite(SpriteConstants.STANDING, SpriteConstants.LEFT)
                else:
                   self.updateSprite(SpriteConstants.STANDING, SpriteConstants.RIGHT)


            # restart enemy's timer from time to time, so that he stops moving.
            if atack3 == 1:
                self.timeenemy = 20
                # puts the enemy turned to the player's direction
                if game.enemymode == 'static':
                    if game.player.rect.right < self.rect.left:
                        self.direction = -1
                    elif game.player.rect.left > self.rect.right:
                        self.direction = 1
                else:
                    self.direction = self.direction * -1

            # checks collision of the player with the enemy
            if self.rect.colliderect(game.player.rect):

                # choses what sprite penalise according to config
                if game.contacthurt == "player":
                    game.player.life = max(0, game.player.life-(game.level*1))
                if game.contacthurt == "enemy":
                    game.enemy.life = max(0, game.enemy.life-(game.level*1))

                # pushes player when he collides with the enemy
                game.player.rect.x +=  self.direction *  50 * dt

                # limits the player to stand on the screem space even being pushed
                if game.player.rect.x < 60:
                    game.player.rect.x = 60
                if game.player.rect.x > 620:
                    game.player.rect.x = 620

                # sets flag to change the player image when he is hurt
                game.player.hurt = 5

            # gravity
            self.dy = min(400, self.dy + 100)
            self.rect.y += self.dy * dt

            # controls screen walls and platforms limits agaist enemy
            new = self.rect
            self.resting = 0
            for cell in game.tilemap.layers['triggers'].collide(new, 'blockers'):

                blockers = cell['blockers']

                if 'l' in blockers and last.right <= cell.left and new.right > cell.left:
                    new.right = cell.left

                if 'r' in blockers and last.left >= cell.right and new.left < cell.right:
                    new.left = cell.right

                if 't' in blockers and last.bottom <= cell.top and new.bottom > cell.top:
                    self.resting = 1
                    new.bottom = cell.top
                    self.dy = 0

                if 'b' in blockers and last.top >= cell.bottom and new.top < cell.bottom:
                    new.top = cell.bottom


            # enemy shoots
            if atack4 == 1:

                self.shooting = 5
                self.gun_cooldown = 3

                # bullets sound effect
                if game.sound == "on" and game.playermode == "human":
                    sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                    c = pygame.mixer.Channel(3)
                    c.set_volume(10)
                    c.play(sound)

                # shoots 4 bullets placed in fixed places - bullets coming from the enemy.
                for i in range (0,4):

                    ay = [-10,-10,20,-45]

                    if self.direction > 0:
                        ax = [-24,50,1,1]
                        self.twists.append(Bullet_e3((self.rect.x+ax[i],self.rect.y-ay[i]), 1, 'h', len(self.twists), game.sprite_e))
                    else:
                        ax = [25,-50,-7,-7]
                        self.twists.append(Bullet_e3((self.rect.x-ax[i],self.rect.y-ay[i]), -1, 'h', len(self.twists), game.sprite_e))

                # shoots 4 bullets placed in fixed places - bullets coming from the top of the screen
                aux = 100
                for i in range (0,4):
                    self.twists.append(Bullet_e3((aux,100), 1, 'v',len(self.twists),game.sprite_e))
                    aux = aux + 150

            # decreases time for bullets limitation
            self.gun_cooldown = max(0, self.gun_cooldown - dt)

            # hurt enemy animation
            if self.hurt > 0:
                if self.direction == -1:
                   self.updateSprite(SpriteConstants.HURTING, SpriteConstants.LEFT)
                else:
                   self.updateSprite(SpriteConstants.HURTING, SpriteConstants.RIGHT)

            self.hurt -=1

            # changes bullets images according to the enemy direction
            if self.shooting > 0:
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.RIGHT)

            self.shooting -= 1
            self.shooting = max(0,self.shooting)


    def updateSprite(self, state, direction):
        self.image = self.spriteDefinition.getImage(state, direction)

# enemy's bullet
class Bullet_e3(pygame.sprite.Sprite):



    image = pygame.image.load('evoman/images/met.png')

    def __init__(self, location, direction, btype, n_twist, *groups):
        super(Bullet_e3, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.lifespan = 100
        self.btype = btype
        self.swingtime = 0
        self.n_twist = n_twist



    def update(self, dt, game):


        if game.time%2==0:
            self.image = pygame.image.load('evoman/images/met.png')
        else:
            self.image = pygame.image.load('evoman/images/met2.png')


        # decreases bullet's timer
        self.lifespan -= 1

        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        # moves the bullets
        if self.btype == 'h':  # bullets that come from the enemy
            if self.lifespan <= 50:
                self.rect.x += self.direction * 550 * dt
        else:
            if self.lifespan <= 60: # bullets that come from the top
                self.rect.y += 300 * dt

                # animation of the bullets swinging
                self.swingtime += 1

                if self.swingtime == 10:
                    self.rect.x += self.direction * 1000 * dt
                    self.direction = self.direction * -1
                    self.swingtime = 0

        # checks collision of enemy's bullet with the player
        if self.rect.colliderect(game.player.rect):

            # player loses life points, accoring to the difficult level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*1))

            # pushes player when he collides with the enemy
            game.player.rect.x +=  self.direction *  100 * dt

            # limits the player to stand on the screen space even being pushed.
            if game.player.rect.x < 60:
                game.player.rect.x = 60
            if game.player.rect.x > 620:
                game.player.rect.x = 620

            # sets flag to change the player image when he is hurt
            game.player.hurt = 5

        # removes player's bullets when colliding with enemy's bullets
        aux = 0
        for t in game.player.twists:
            if t != None:
                if self.rect.colliderect(t.rect):
                    t.kill()
                    game.player.twists[aux] = None
            aux += 1
