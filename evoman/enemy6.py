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
timeexpire = 2200 # game run limit

# enemy 6 sprite, crashman
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
        self.twists = []
        self.alternate = 1
        self.just_shoot = 0
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


        # Defines game mode for player action.
        if game.enemymode == 'static': # Enemy controlled by static movements.

            if self.timeenemy == 105:
                atack1 = 1
            else:
                atack1 = 0

            if ( (abs(self.rect.left-game.player.rect.left)<=1 or abs(self.rect.right-game.player.rect.right)<=1)  or self.dy>200  ):
                atack2 = 1
            else:
                atack2 = 0


            atack3 = 0


        elif game.enemymode == 'ai': # Player controlled by AI algorithm.


            # calls the controller providing game sensors
            actions = game.enemy_controller.control(self.sensors.get(game), game.econt)
            if len(actions) < 3:
                game.print_logs("ERROR: Enemy 1 controller must return 3 decision variables.")
                sys.exit(0)

            atack1 = actions[0]
            atack2 = actions[1]
            atack3 = actions[2]



            if atack2 == 1 and not self.gun_cooldown:
                atack2 = 1
            else:
                atack2 = 0


        # if the 'start game' marker is 1
        if game.start == 1:

            self.timeenemy += 1 # increments enemy timer

            last = self.rect.copy() # copies last position state of the enemy

            # movements of the enemy on the axis x until a limit (turning aroud)

            if self.rect.left<60:
                self.direction = self.direction  * -1
                self.rect.left = 60
            if self.rect.right>680:
                self.direction = self.direction  * -1
                self.rect.right = 680

            # calculating the relative distance between enemy and player to set the jumping strengh
            aux_dist = (abs(game.player.rect.right - self.rect.right)/490.0)+0.1

            # when atacking, enemy may accelarate his movement.
            if self.dy<0:
                self.rect.x += self.direction * (1500 *aux_dist) * dt
            else:
                self.rect.x += self.direction * 180 * dt

            #  jumps over the player. It happens from time to time, or when the player shoots.
            if ((self.resting == 1 and atack1 == 1) or ( self.resting == 1 and game.player.atacked == 1)):

                if game.enemymode == 'static':
                    # enemy turns to the players direction.
                    if game.player.rect.right <= self.rect.left:
                        self.direction = -1
                    if game.player.rect.left >= self.rect.right:
                        self.direction = 1

                # reinicializes enemy timer
                self.timeenemy = 0

                self.dy = -1500 * aux_dist
                self.resting = 0

            if atack3 == 1 and game.enemymode == 'ai':
                 self.direction = self.direction * -1

            # throws a bullet over the player when enemy is jumping and right over him
            if self.resting == 0 and self.just_shoot == 0 and atack2 == 1:

                self.shooting = 5
                self.gun_cooldown = 3

                # bullets sound effect
                if game.sound == "on" and game.playermode == "human":
                    sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                    c = pygame.mixer.Channel(3)
                    c.set_volume(10)
                    c.play(sound)

                self.just_shoot = 1

                self.twists.append(Bullet_e6((self.rect.x ,self.rect.y ), self.direction, len(self.twists), game.sprite_e))


            self.gun_cooldown = max(0, self.gun_cooldown - dt) # decreases time for bullets limitation

            # animation, running enemy images alternation
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

                game.player.hurt = 5  # sets flag to change the player image when he is hurt

            #  gravity
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
                    self.just_shoot = 0

                if 'b' in blockers and last.top >= cell.bottom and new.top < cell.bottom:
                    new.top = cell.bottom

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
class Bullet_e6(pygame.sprite.Sprite):



    image = pygame.image.load('evoman/images/mi2.png')

    def __init__(self, location, direction, n_twist, *groups):
        super(Bullet_e6, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.lifespan = 70
        self.n_twist = n_twist


    def update(self, dt, game):


        self.rect.y +=  500 * dt  # moves the bullets

        self.rect.y = min(410,self.rect.y) # prevents bullets from passing throught the floor

        self.lifespan -= 1 #  decreases bullet's timer

        # removes old bullets
        if self.lifespan < 0:
            self.kill()
            game.enemy.twists[self.n_twist] = None
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
