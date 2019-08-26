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

tilemap = 'evoman/map4.tmx'
timeexpire = 1000 # game run limit

# enemy 7 sprite, bubbleman
class Enemy(pygame.sprite.Sprite):


    def __init__(self, location, *groups):
        super(Enemy, self).__init__(*groups)
        self.spriteDefinition = SpriteDefinition('evoman/images/EnemySprites.png', 0, 0, 43, 59)
        self.updateSprite(SpriteConstants.STANDING, SpriteConstants.LEFT)

        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = -1
        self.max_life = 100
        self.life = self.max_life
        self.resting = 0
        self.dy = 0
        self.alternate = 1
        self.imune = 0
        self.timeenemy = 0
        self.twists = []
        self.bullets = 0
        self.hurt = 0
        self.shooting = 0
        self.gun_cooldown = 0
        self.gun_cooldown2 = 0



    def update(self, dt, game):

        if game.time==1:
            # puts enemy in random initial position
            if game.randomini == 'yes':
                self.rect.x = numpy.random.choice([640,500,400,300])


        # defines game mode for player actionv
        if game.enemymode == 'static': # enemy controlled by static movements

            if self.timeenemy>=4 and self.timeenemy<=20 and  self.timeenemy%4 == 0:
                atack1 = 1
            else:
                atack1 = 0


            atack2 = 1 #useless


            if self.timeenemy==4:
                atack2 = 1
            else:
                atack2 = 0

            if self.timeenemy==5:
                atack3 = 1
            else:
                atack3 = 0

            if self.timeenemy>=50 and  self.timeenemy<80:
                atack4 = 1
            else:
                atack4 = 0

            if self.timeenemy == 50:
                atack5 = 1
            else:
                atack5 = 0

            if self.timeenemy == 100:
                atack6 = 1
            else:
                atack6 = 0


        elif game.enemymode == 'ai': # Player controlled by AI algorithm.



            # calls the controller providing game sensors
            actions = game.enemy_controller.control(self.sensors.get(game), game.econt)  
            if len(actions) < 6:
                game.print_logs("ERROR: Enemy 1 controller must return 6 decision variables.")
                sys.exit(0)

            atack1 = actions[0]
            atack2 = actions[1]
            atack3 = actions[2]
            atack4 = actions[3]
            atack5 = actions[4]
            atack6 = actions[5]


            if atack1 == 1 and not self.gun_cooldown2:
                atack1 = 1
            else:
                atack1 = 0


            if atack3 == 1 and not self.gun_cooldown:
                atack3 = 1
            else:
                atack3 = 0


        # marks the flag indicating to the player that the map is on water environment
        game.player.inwater = 1

        # if the 'start game' marker is 1
        if game.start == 1:

            # increments enemy timer
            self.timeenemy += 1

            # copies last position state of the enemy
            last = self.rect.copy()

            # calculates a relative distance factor, between the player and enemy to set up the jumping strengh
            aux_dist = (abs(game.player.rect.right - self.rect.right)/490.0)+0.3

            # shoots 5 bullets positioned over the same range
            if atack1 == 1:

                self.shooting = 5

                self.gun_cooldown2 = 3


                # bullets sound effect
                if game.sound == "on" and game.playermode == "human":
                    sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                    c = pygame.mixer.Channel(3)
                    c.set_volume(10)
                    c.play(sound)

                rand = numpy.random.randint(0, 25, 1)
                self.twists.append(Bullet_e7((self.rect.x,self.rect.y), self.direction, len(self.twists), game.sprite_e))


            # throws from 1 to 3 bubbles, starting at slighly different positions

            if self.bullets == 0: # if the bubblues have gone away, enemy is abble to realease new bubbles.

                rand = 2
                for i in range(0,rand):
                    if atack3 == 1:

                        self.gun_cooldown = 3

                        self.bullets += 1
                        self.twists.append(Bullet_e72((self.rect.x+self.direction*i*30  ,self.rect.y-i*30), self.direction, len(self.twists), game.sprite_e))

            # decreases time for bullets limitation
            self.gun_cooldown = max(0, self.gun_cooldown - dt)

            # decreases time for bullets limitation
            self.gun_cooldown2 = max(0, self.gun_cooldown2 - dt)

            # enemy moves during some time, after standing still for a while
            if atack4 == 1:
                   self.rect.x += self.direction * 600 * aux_dist * dt * 0.7

            #  enemy jumps while is moving
            if self.resting == 1 and atack5 == 1:
                self.dy = -1500
                self.resting = 0

            # at the end of the atack cicle, enemy turns over the players direction.
            if atack6 == 1:
               if game.enemymode == 'static':
                   if game.player.rect.right < self.rect.left:
                       self.direction = -1
                   if game.player.rect.left > self.rect.right:
                        self.direction = 1
               else:
                   self.direction = self.direction * -1

               # reinicializes enemy timer
               self.timeenemy = 0

            #  gravity
            self.dy = min(400, self.dy + 100)
            self.rect.y += self.dy * dt * 0.4

            #  changes the image when enemy jumps or stands up
            if self.resting == 0:
               if self.direction == -1:
                   self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.LEFT)
               else:
                   self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.RIGHT)
            else:
               if self.direction == -1:
                   self.updateSprite(SpriteConstants.STANDING, SpriteConstants.LEFT)
               else:
                   self.updateSprite(SpriteConstants.STANDING, SpriteConstants.RIGHT)

            # checks collision of the player with the enemy
            if self.rect.colliderect(game.player.rect):

                # choses what sprite penalise according to config
                if game.contacthurt == "player":
                    game.player.life = max(0, game.player.life-(game.level*0.3))
                if game.contacthurt == "enemy":
                    game.enemy.life = max(0, game.enemy.life-(game.level*0.3))


                game.player.rect.x +=  self.direction *  50 * dt   # pushes player when he collides with the enemy

                # limits the player to stand on the screen space even being pushed
                if game.player.rect.x < 60:
                    game.player.rect.x = 60
                if game.player.rect.x > 620:
                    game.player.rect.x = 620

                if self.rect.x < 70:
                    self.rect.x = 70
                if self.rect.x > 610:
                    self.rect.x = 610


                game.player.hurt = 5 # Sets flag to change the player image when he is hurt.

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

            # hurt enemy image
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
class Bullet_e7(pygame.sprite.Sprite):

    image = pygame.image.load('evoman/images/bullet2_l.png')

    def __init__(self, location, direction, n_twist, *groups):
        super(Bullet_e7, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.n_twist = n_twist



    def update(self, dt, game):


        self.rect.x +=  self.direction * 500 * dt  # moves the bullets on the axis x

        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        # checks collision of enemy's bullet with the player
        if self.rect.colliderect(game.player.rect):

            # player loses life points, accoring to the difficult level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*0.3))


            game.player.rect.x +=  self.direction *  100 * dt  # pushes player when he collides with the enemy

            # limits the player to stand on the screen space even being pushed
            if game.player.rect.x < 60:
                game.player.rect.x = 60
            if game.player.rect.x > 620:
                game.player.rect.x = 620

            # sets flag to change the player image when he is hurt
            game.player.hurt = 1
        else:
            game.player.hurt = 0



# enemy's bullet 2 (bubble)
class Bullet_e72(pygame.sprite.Sprite):

    image = pygame.image.load('evoman/images/bubb.png')

    def __init__(self, location, direction, n_twist, *groups):
        super(Bullet_e72, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.direc = 1
        self.n_twist = n_twist



    def update(self, dt, game):


        self.rect.x +=  self.direction * 200 * dt * 0.5      # moves the bullets on the axis x

        # moves the bullets on the axis y. Go up and down according to the floor and imaginary top.
        self.rect.y += 200 * self.direc * dt * 0.4
        if self.rect.y >= 460 or self.rect.y <= 350:
            self.direc = self.direc * -1

        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            game.enemy.bullets -=1
            return

        # checks collision of enemy's bullet with the player
        if self.rect.colliderect(game.player.rect):

            # player loses life points, according to the difficulty level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*0.3))

            game.player.rect.x +=  self.direction *  100 * dt # pushes player when he collides with the enemy

            # limits the player to stand on the screen space even being pushed
            if game.player.rect.x < 60:
                game.player.rect.x = 60
            if game.player.rect.x > 620:
                game.player.rect.x = 620


            game.player.hurt = 5 # sets flag to change the player image when he is hurt
