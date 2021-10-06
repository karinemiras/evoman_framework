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
timeexpire = 1500 # game run limit

# enemy 4 sprite, heatman
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
        self.fireflash = 0
        self.imune = 0
        self.rect.x = 550
        self.timeenemy = 0
        self.hurt = 0
        self.shooting = 0
        self.gun_cooldown = 0
        self.rect.right = 580


    def update(self, dt, game):


        if game.time==1:
            # puts enemy in random initial position
            if game.randomini == 'yes':
                self.rect.x = numpy.random.choice([640,500,400,300])


        # defines game mode for player action
        if game.enemymode == 'static': # enemy controlled by static movements

            if self.timeenemy == 2:
                atack1 = 1
            else:
                atack1 = 0

            if self.timeenemy> 50:
                atack2 = 1
            else:
                atack2 = 0

            if self.timeenemy == 3:
                atack3 = 1
            else:
                atack3 = 0

            if (self.fireflash>=1 and self.fireflash <=40):
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

            if atack3 == 1 and not self.gun_cooldown:
                atack3 = 1
            else:
                atack3 = 0



        # if the 'start game' marker is 1
        if game.start == 1:

            self.timeenemy += 1 # increments enemy timer

            last = self.rect.copy()  # copies last position state of the enemy


            # when player atacks, enemy turns into fire and goes towards his direction
            if game.player.atacked == 1 and self.fireflash == 0:
                self.fireflash = 100
            else:
                self.fireflash = max(0,self.fireflash -1)

            if  atack4 == 1:
                self.rect.x += self.direction * 600 * dt

                if self.fireflash == 1:
                    self.direction = self.direction * -1

                if  self.rect.colliderect(game.player.rect):
                    self.fireflash = 0

            # otherwise he just keeps shooting towards the player direction
            elif self.fireflash == 0:

                if atack1 == 1 and self.resting == 1:
                    self.dy = -900
                    self.resting = 0

                self.imune = 0  # enemy is not imune to player's shooting anymore

                # images of the enemy standing up
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.STANDING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.STANDING, SpriteConstants.RIGHT)

            # reinicializes timer and turns to the players direction
            if atack2 == 1:
                self.timeenemy = 1

                if game.enemymode == 'static':
                    if game.player.rect.right < self.rect.left:
                        self.direction = -1
                    elif game.player.rect.left > self.rect.right:
                        self.direction = 1
                else:
                   self.direction = self.direction *-1


            # checks collision of the player with the enemy
            if self.rect.colliderect(game.player.rect):

                # choses what sprite penalise according to config
                if game.contacthurt == "player":
                    game.player.life = max(0, game.player.life-(game.level*0.3))
                if game.contacthurt == "enemy":
                    game.enemy.life = max(0, game.enemy.life-(game.level*0.3))

                # pushes player when he collides with the enemy
                game.player.rect.x +=  self.direction *  50 * dt

                # limits the player to stand on the screen space even being pushed
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


            # enemy shoots 3 bullets
            if atack3 == 1:

                self.shooting = 5

                self.gun_cooldown = 5

                # if enemy is not turned into fire, shoots, otherwise stops the time counter for a while.
                if self.fireflash == 0:

                    # bullets sound effect
                    if game.sound == "on" and game.playermode == "human":
                        sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                        c = pygame.mixer.Channel(3)
                        c.set_volume(10)
                        c.play(sound)


                    for i in range (0,3):
                        self.twists.append(Bullet_e4((self.rect.x ,self.rect.y ), self.direction, i, len(self.twists), game.sprite_e))
                else :
                    self.timeenemy -= 1


            self.gun_cooldown = max(0, self.gun_cooldown - dt)  # decreases time for bullets limitation.

           # changes bullets images according to the enemy direction
            if self.shooting > 0:
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.RIGHT)

            self.shooting -= 1
            self.shooting = max(0,self.shooting)

            #  changes the image when enemy is hurt and imune, as a fireball
            if self.imune == 1:
                if game.time%2==0:
                    self.image = pygame.image.load('evoman/images/fireball.png')
                else:
                    self.image = pygame.image.load('evoman/images/fireball2.png')

            self.hurt -=1

    def updateSprite(self, state, direction):
        self.image = self.spriteDefinition.getImage(state, direction)


# enemy bullets
class Bullet_e4(pygame.sprite.Sprite):

    image = pygame.image.load('evoman/images/bullet_l.png')

    def __init__(self, location, direction, n, n_twist, *groups):
        super(Bullet_e4, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.lifespan = 30
        self.n= n
        self.n_twist = n_twist

    def update(self, dt, game):


        # puts the bullets in positions relative to the player. They go from the enemy to where the player is.
        if self.n == 0:
            aux_x = 50
            aux_y = (abs(game.player.rect.x - game.enemy.rect.x)*0.55)
        elif self.n == 1:
            aux_x = 20
            aux_y = (abs(game.player.rect.x - game.enemy.rect.x)*0.60)
        elif self.n == 2:
            aux_x = -10
            aux_y = (abs(game.player.rect.x - game.enemy.rect.x)*0.65)

        # bullets axis x movement
        if self.direction == -1:
            if self.rect.x > game.player.rect.left + aux_x:
                self.rect.x += self.direction *  650  * dt
        else:
            if self.rect.x < game.player.rect.right - aux_x:
                self.rect.x += self.direction *  650  * dt

        # bullets axis y movements
        if self.direction == -1:
             if self.rect.x > game.player.rect.left + aux_y:
                 self.rect.y -=  500 * dt
             else:
                 self.rect.y +=  700 * dt
        else:
             if self.rect.x < game.player.rect.right - aux_y-10:
                 self.rect.y -=  500 * dt
             else:
                 self.rect.y +=  700 * dt

        # prevents bullets from passing through the floor
        self.rect.y = min(410,self.rect.y)

        # removes old bullets
        if self.rect.y == 410:
            self.lifespan -= 1

        if self.lifespan < 0:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        if self.rect.right<1 or self.rect.left>736 or  self.rect.top <1 or self.rect.bottom>512 :
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        # checks collision of enemy's bullet with the player
        if self.rect.colliderect(game.player.rect):

            # player loses life points, accoring to the difficulty level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*0.3))

            # pushes player when he collides with the enemy
            game.player.rect.x +=  self.direction *  100 * dt

           # limits the player to stand on the screen space even being pushed
            if game.player.rect.x < 60:
                game.player.rect.x = 60
            if game.player.rect.x > 620:
                game.player.rect.x = 620

            # sets flag to change the player image when he is hurt
            game.player.hurt = 5
