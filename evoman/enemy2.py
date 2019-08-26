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

# enemy 2 sprite, airman
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
        self.imune = 0
        self.timeenemy = 0
        self.hurt = 0
        self.shooting = 0
        self.gun_cooldown = 0


    def update(self, dt, game):



        if game.time==1:
            # puts enemy in random initial position
            if game.randomini == 'yes':
                self.rect.x = numpy.random.choice([630,610,560,530])


        # defines game mode for player action.
        if game.enemymode == 'static': # enemy controlled by static movements

            if (self.timeenemy >= 210  and self.timeenemy <= 250) or (self.timeenemy >= 260  and self.timeenemy <= 300):
                atack1 = 1
            else:
                atack1 = 0

            if self.timeenemy == 210 or self.timeenemy == 260:
                atack2 = 1
            else:
                atack2 = 0

            if self.timeenemy> 300:
                atack3 = 1
            else:
                atack3 = 0

            if (self.timeenemy == 40) or (self.timeenemy == 110) or (self.timeenemy == 180):
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

            if atack1 == 1 and self.resting == 1:
                atack1 = 1
            else:
                atack1 = 0


        # if the 'start game' marker is 1
        if game.start == 1:

            # increments enemy timer
            self.timeenemy += 1

            # copies last position state of the enemy
            last = self.rect.copy()

            # movements of the enemy on the axis x. Happens 2 to each side.
            if atack1 == 1  :
                self.rect.x += self.direction * 200 * dt

                # jumps
                if atack2 == 1:
                    self.dy = -900
                    self.resting = 0

               # animation, running enemy images alternatetion.
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

            # restart enemy timer and turns the enemy around
            if atack3 == 1:
                self.timeenemy = 0
                self.direction = self.direction * -1

            # checks collision of the player with the enemy
            if self.rect.colliderect(game.player.rect):

                # choses what sprite penalise according to config
                if game.contacthurt == "player":
                    game.player.life = max(0, game.player.life-(game.level*1))
                if game.contacthurt == "enemy":
                    game.enemy.life = max(0, game.enemy.life-(game.level*1))

                game.player.hurt = 5 # sets flag to change the player image when he is hurt.

            # gravity
            self.dy = min(400, self.dy + 100)
            self.rect.y += self.dy * dt

            # controls screen walls and platforms limits agaist enemy.
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


                # shoots 6 bullets placed in a fixed range
                for i in range (0,6):
                    self.twists.append(Bullet_e2((self.rect.x+10,self.rect.bottom), self.direction ,i, len(self.twists), game.sprite_e))

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
class Bullet_e2(pygame.sprite.Sprite):


    image = pygame.image.load('evoman/images/torna.png')

    def __init__(self, location, direction,n, n_twist , *groups):
        super(Bullet_e2, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.lifespan = 55
        self.n = n
        self.n_twist = n_twist



    def update(self, dt, game):

        if game.time%2==0:
            self.image = pygame.image.load('evoman/images/torna.png')
        else:
            self.image = pygame.image.load('evoman/images/torna2.png')


        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
            self.kill()
            game.enemy.twists[self.n_twist] = None
            return

        # enemy atack: blows the player forward with the bullets
        if self.lifespan > 43:
            ax = [100,380,440,270,220,300]
            ay = [30,70,120,-40	,80,130]

            if self.direction == -1:
                if  self.rect.x >= game.enemy.rect.x - ax[self.n]:
                    self.rect.x -= 1400 * dt
            if self.direction == 1:
                if  self.rect.x <= game.enemy.rect.x + ax[self.n]:
                    self.rect.x += 1400	 * dt

            if  self.rect.y >= game.enemy.rect.y - ay[self.n]:
                self.rect.y -= 550 * dt

        elif self.lifespan <= 5:
            self.rect.x += self.direction * 650 * dt
            game.player.rect.x +=  self.direction *  150 * dt

            # limitates player in the screen.
            if game.player.rect.x < 60:
                game.player.rect.x = 60
            if game.player.rect.x > 620:
                game.player.rect.x = 620

        # decreases bullet's timer
        self.lifespan -= 1

        # checks collision of enemy's bullet with the player
        if self.rect.colliderect(game.player.rect):

            # player loses life points, according to the difficult level of the game (the more difficult, the more it loses).
            game.player.life = max(0, game.player.life-(game.level*1))

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
