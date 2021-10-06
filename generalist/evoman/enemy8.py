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

tilemap = 'evoman/map3.tmx'
timeexpire = 1000 # game run limit

# enemy 8 sprite, quickman
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
        self.just_shoot = 0
        self.imune = 0
        self.timeenemy = 0
        self.twists = []
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

            if (self.timeenemy >= 1 and self.timeenemy <10)  or (self.timeenemy >= 20 and self.timeenemy <30):
                atack1 = 1
            else:
                atack1 = 0

            if self.timeenemy == 1  or self.timeenemy == 20:
                atack2 = 1
            else:
                atack2 = 0

            if self.timeenemy == 9  or self.timeenemy == 29:
                atack3 = 1
            else:
                atack3 = 0

            if self.timeenemy >=40 and self.timeenemy <50:
                atack4 = 1
            else:
                atack4 = 0

            if self.timeenemy  == 50:
                atack5 = 1
            else:
                atack5 = 0

            if ( (abs(self.rect.left-game.player.rect.left)<=200 or abs(self.rect.right-game.player.rect.right)<=200)  ) and not self.gun_cooldown:
                atack6 = 1
            else:
                atack6 = 0



        elif game.enemymode == 'ai': # player controlled by AI algorithm



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



            if atack6 == 1 and not self.gun_cooldown:
                atack6 = 1
            else:
                atack6 = 0


        # If the 'start game' marker is 1.
        if game.start == 1:

            self.timeenemy += 1 # increments enemy timer

            last = self.rect.copy() # copies last position state of the enemy

            # jumps over the player. It happens twice.
            if  atack1== 1 :
                # moves on the axis x
                self.rect.x += self.direction * 730 * dt
                if self.resting == 1 and atack2 == 1:
                    self.dy = -900
                    self.resting = 0

                # enemy turns to the players direction
                if atack3 == 1:

                    if game.enemymode == 'static':
                        if game.player.rect.right < self.rect.left:
                            self.direction = -1
                        if game.player.rect.left > self.rect.right:
                            self.direction = 1
                    else:
                        self.direction = self.direction * -1

            # runs in the player's direction, after jumping twice
            elif atack4 == 1 :
                self.rect.x += self.direction * 900 * dt

            # reinicializes enemy timer
            elif atack5 == 1:
                self.timeenemy = 0

            # throws a bullet over the player when enemy is jumping and right over him
            if self.resting == 0 and self.just_shoot == 0 and atack6 == 1:

                self.shooting = 5

                self.gun_cooldown = 5

                # bullets sound effect
                if game.sound == "on" and game.playermode == "human":
                    sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                    c = pygame.mixer.Channel(3)
                    c.set_volume(10)
                    c.play(sound)

                self.just_shoot = 1

                rand = 3
                # shoots from 1 to 3 bullets
                for i in range(0,rand):
                    self.twists.append(Bullet_e8((self.rect.x+(i*60) ,self.rect.y ), i, self.direction, len(self.twists), game.sprite_e))

            # decreases time for bullets limitation
            self.gun_cooldown = max(0, self.gun_cooldown - dt)

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

                # sets flag to change the player image when he is hurt
                game.player.hurt = 5

            #  gravity
            self.dy = min(400, self.dy + 100)
            self.rect.y += self.dy * dt

            # controls screen walls and platforms limits agaist enemy
            new = self.rect
            self.resting = 0
            for cell in game.tilemap.layers['triggers'].collide(new, 'blockers'):

                blockers = cell['blockers']

                if 't' in blockers and last.bottom <= cell.top and new.bottom > cell.top:
                    self.resting = 1
                    new.bottom = cell.top
                    self.dy = 0
                    self.just_shoot = 0

                if 'b' in blockers and last.top >= cell.bottom and new.top < cell.bottom:
                    new.top = cell.bottom

                if 'l' in blockers and last.right <= cell.left and new.right > cell.left  and last.bottom>cell.top:
                    new.right = cell.left

                if 'r' in blockers and last.left >= cell.right and new.left < cell.right   and last.bottom>cell.top:
                    new.left = cell.right

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
class Bullet_e8(pygame.sprite.Sprite):

    image = pygame.image.load('evoman/images/bullet2_l.png')

    def __init__(self, location, direction, n, n_twist, *groups):
        super(Bullet_e8, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.lifespan = 70
        self.n = n
        self.n_twist = n_twist



    def update(self, dt, game):


        self.lifespan -= 1 # decreases bullet's timer


        # moves the bullets up after sometime
        if self.lifespan < 40:
            self.rect.y -=  700 * dt
        else:
            self.rect.y +=  500 * dt # moves the bullets down when it is shoot
            self.rect.y = min(410,self.rect.y) # preevens bullets from going away


        # moves the bullet on the axis x according to the player's direction
        if not (abs(self.rect.left-game.player.rect.left)<=10 or abs(self.rect.right-game.player.rect.right)<=10):
            if game.player.rect.left < self.rect.left:
               self.rect.x -= (400) * dt
            else:
               self.rect.x += (400) * dt

        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        # checks collision of enemy's bullet with the player
        if self.rect.colliderect(game.player.rect):

            # player loses life points, according to the difficult level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*0.3))

            game.player.hurt = 5 # sets flag to change the player image when he is hurt
