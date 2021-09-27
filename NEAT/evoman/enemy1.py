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

tilemap = 'evoman/map1.tmx'  # scenario
timeexpire = 1000 # game run limit


# enemy 1 sprite, flashman
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
        self.time_colis = 0
        self.alternate = 1
        self.imune = 0
        self.timeenemy = 0
        self.twists = []
        self.hurt = 0
        self.shooting = 1
        self.gun_cooldown = 0
        self.gun_cooldown2 = 0




    def update(self, dt, game):


        if game.time==1:
            # puts enemy in random initial position
            if game.randomini == 'yes':
                self.rect.x = numpy.random.choice([640,500,400,300])


        # increments enemy timer
        if game.start == 1:
            self.timeenemy += 1

        # defines game mode for player action
        if game.enemymode == 'static': #  controlled by static movements

            atack1 = 1

            if self.timeenemy >= 200 and self.timeenemy < 260:
                atack2 = 1
            else:
                atack2 = 0

            if self.timeenemy == 220:
                atack3 = 1
            else:
                atack3 = 0

            atack4 = 1


        elif game.enemymode == 'ai': # enemy controlled by AI algorithm


            # calls the controller providing game sensors
            actions = game.enemy_controller.control(self.sensors.get(game), game.econt)
            if len(actions) < 4:
                game.print_logs("ERROR: Enemy 1 controller must return 4 decision variables.")
                sys.exit(0)

            atack1 = actions[0]
            atack2 = actions[1]
            atack3 = actions[2]
            atack4 = actions[3]


            # applies attack rules
            if atack2 == 1 and not self.gun_cooldown:
                atack2 = 1
            else:
                atack2 = 0

            if atack3 == 1 and not self.gun_cooldown2:
                atack3 = 1
            else:
                atack3 = 0



        # if the enemy is not atacking with the feezing atack (prevents player from making any movements) and also the 'start game' marker is 1.
        if game.freeze_e == 0 and game.start == 1:

            last = self.rect.copy()# copies last position state of the enemy

            if atack1 == 1:
                # moves the enemy on the axis x
                self.rect.x += self.direction * 100 * dt

                # chases player, switching direction as he moves.
                if atack4 == 1:

                    if game.enemymode == 'static':
                        if game.player.rect.right < self.rect.left:
                            self.direction = -1
                        elif game.player.rect.left > self.rect.right:
                             self.direction = 1
                    else:
                        self.direction = self.direction * -1

            # animation, running enemy images alternation.
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

            # checks collision of the player with the enemy
            if self.rect.colliderect(game.player.rect):

                # sprite loses life points, according to the difficult level of the game (the more difficult, the more it loses).

                # choses what sprite penalise according to config
                if game.contacthurt == "player":
                    game.player.life = max(0, game.player.life-(game.level*1))
                if game.contacthurt == "enemy":
                    game.enemy.life = max(0, game.enemy.life-(game.level*1))

                # counts duration of the collision to jump from time to time during the collision
                self.time_colis += 1
                if self.time_colis > 15:
                    self.time_colis = 0
                    self.dy = -600

                # sets flag to change the player image when he is hurt
                game.player.hurt = 5


            # gravity
            self.dy = min(400, self.dy + 100)
            self.rect.y += self.dy * dt

            # controls screen walls and platforms limits towards enemy
            new = self.rect
            self.resting = 0
            for cell in game.tilemap.layers['triggers'].collide(new, 'blockers'):

                blockers = cell['blockers']

                if 't' in blockers and last.bottom <= cell.top and new.bottom > cell.top:
                    self.resting = 1
                    new.bottom = cell.top
                    self.dy = 0

                if 'b' in blockers and last.top >= cell.bottom and new.top < cell.bottom:
                    new.top = cell.bottom

                if 'l' in blockers and last.right <= cell.left and new.right > cell.left  and last.bottom>cell.top:
                    new.right = cell.left
                    # Jumps when finds a wall in the middle plataforms.
                    if new.left<600:
                        self.dy = -600

                if 'r' in blockers and last.left >= cell.right and new.left < cell.right and last.bottom>cell.top:
                    new.left = cell.right
                    # Jumps when finds a wall in the middle plataforms.
                    if new.left>29:
                        self.dy = -600

            #  Changes the image when enemy jumps.
            if self.resting == 0:
               if self.direction == -1:
                   self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.LEFT)
               else:
                   self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.RIGHT)

            # Hurt enemy animation.
            if self.hurt > 0:
                if self.direction == -1:
                   self.updateSprite(SpriteConstants.HURTING, SpriteConstants.LEFT)
                else:
                   self.updateSprite(SpriteConstants.HURTING, SpriteConstants.RIGHT)

            self.hurt -=1


        # Enemy atack: freezes the player (preeveting him from making any movements or atacking) and also himself from moving. Freenzing endures according to the timer.
        if atack2 == 1:

            self.gun_cooldown = 6

            game.freeze_p = 1
            game.freeze_e = 1

        # Enemy shooting after freezing.
        if  atack3 == 1:

            self.shooting = 5

            self.gun_cooldown2 = 6

            # Bullets sound effect.
            if game.sound == "on" and game.playermode == "human":

                sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                c = pygame.mixer.Channel(3)
                c.set_volume(10)
                c.play(sound)

            # Shoots 8 bullets placed in a fixed range with a little random variation in their position (x and y).


            for i in range (0,8):
                rand = numpy.array([30,20,10,15,9,25,18,5])
                rand2 = numpy.array([1,2,3,4,5,2,4,3])

                rand = rand[i]
                rand2 = rand2[i]

                # Start position of the bullets vary according to the position of the enemy.
                if self.direction > 0:
                    self.twists.append(Bullet_e1((self.rect.x+(i*rand),self.rect.y+10+(i*rand2)), 1, len(self.twists), game.sprite_e))
                else:
                    self.twists.append(Bullet_e1((self.rect.x-(i*rand)+46,self.rect.y+10+(i*rand2)), -1, len(self.twists), game.sprite_e))

        # Decreases time for bullets and freezing limitation.
        self.gun_cooldown = max(0, self.gun_cooldown - dt)
        self.gun_cooldown2 = max(0, self.gun_cooldown2 - dt)

        # Changes bullets images according to the enemy direction.
        if self.shooting > 0:
            if self.direction == -1:
                self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.LEFT)
            else:
                self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.RIGHT)

        self.shooting -= 1
        self.shooting = max(0,self.shooting)

        # Releases movement.
        if  self.gun_cooldown <= 5:
            game.freeze_p = 0
            game.freeze_e = 0

        # Reinicializes enemy atacking timer.
        if self.timeenemy == 260:
            self.timeenemy = 0

    def updateSprite(self, state, direction):
        self.image = self.spriteDefinition.getImage(state, direction)

# Enemy's bullets.
class Bullet_e1(pygame.sprite.Sprite):



    image = pygame.image.load('evoman/images/bullet2_l.png')

    def __init__(self, location, direction, n_twist, *groups):
        super(Bullet_e1, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.n_twist = n_twist

        # Fits image according to the side the enemy is turned to.
        if self.direction == 1:
            self.image = pygame.image.load('evoman/images/bullet2_r.png')
        else:
            self.image = pygame.image.load('evoman/images/bullet2_l.png')



    def update(self, dt, game):

        # Removes bullets objetcs when they transpass the screem limits.
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        # Moving on the X axis.
        self.rect.x += self.direction * 300 * dt

        # Checks collision of enemy's bullet with the player.
        if self.rect.colliderect(game.player.rect):

            # Player loses life points, accoring to the difficult level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*3))

            # Removes the bullet off the screem after collision.
            self.kill()
            game.enemy.twists[self.n_twist] = None

            # Sets flag to change the player image when he is hurt.
            game.player.hurt = 5
