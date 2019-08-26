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

# enemy 5 sprite, metalman
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
        self.alternate = 1
        self.direction_floor = 1
        self.imune = 0
        self.move = 0
        self.countmove = 0
        self.rect.x = 500
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

        # Defines game mode for player action.
        if game.enemymode == 'static': # Enemy controlled by static movements.

            if self.resting == 1 and self.timeenemy >= 95 and self.timeenemy <= 110:
                atack1 = 1
            else:
                atack1 = 0

            if self.resting == 0 :
                atack2 = 1
            else:
                atack2 = 0

            if (game.player.rect.right < game.enemy.rect.left and  abs(game.player.rect.right - game.enemy.rect.left) <= 50 ) or (game.enemy.rect.right < game.player.rect.left and abs(game.enemy.rect.right - game.player.rect.left) <= 50):
                atack3 = 1
            else:
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

            # moving floor, changes the movement direction from time to time.
            for cell in game.tilemap.layers['triggers'].collide(game.player.rect, 'blockers'):

                blockers = cell['blockers']
                if 't' in blockers:

                    game.player.rect.x += self.direction_floor * 100 * dt # moves player over the moving floor

                    # limits player inside the screen
                    if game.player.rect.left < 60:
                        game.player.rect.left = 61
                    if game.player.rect.right > 665:
                        game.player.rect.right = 665

                if game.time%120 == 0:
                    self.direction_floor = self.direction_floor *-1


            last = self.rect.copy() # copies last position state of the enemy

            # if player gets too close to the enemy, he jumps to the other side.
            if (self.resting == 1 and atack3 == 1):
                self.move = 1
                self.dy = -900
                self.resting = 0

            if self.move == 1:
                self.rect.x += self.direction * 900 * dt

            if self.move == 1 and self.rect.x<200:
                self.rect.x = 200
                self.direction = self.direction  * -1
                self.move = 0
            if self.move == 1 and  self.rect.x>500:
                self.rect.x = 500
                self.direction = self.direction  * -1
                self.move = 0


            # jumps from time to time or when player atacks
            if ( (self.resting == 1 and atack1 == 1) or ( self.resting == 1 and game.player.atacked == 1)):
                self.dy = -900
                self.resting = 0

            # releases until 4 bullets (decided randomly) after jumping or when player atacks
            if (atack2 == 1 and not self.gun_cooldown) :

                self.shooting = 5

                self.gun_cooldown = 3

                # bullets sound effect
                if game.sound == "on" and game.playermode == "human":
                    sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                    c = pygame.mixer.Channel(3)
                    c.set_volume(10)
                    c.play(sound)

                aux = numpy.random.randint(1,4)
                for i in range(0,aux):
                    self.twists.append(Bullet_e5((self.rect.x + (self.direction*(i*30)) ,self.rect.top + (self.direction*(i*20))  ), self.direction, game.player.rect , len(self.twists), game.sprite_e))


                self.timeenemy = 0 # reinicializes enemy timer

            if game.player.atacked == 1:
                # bullets sound effect
                if game.sound == "on" and game.playermode == "human":
                    sound = pygame.mixer.Sound('evoman/sounds/scifi011.wav')
                    c = pygame.mixer.Channel(3)
                    c.set_volume(10)
                    c.play(sound)

                self.twists.append(Bullet_e5((self.rect.x ,self.rect.top ), self.direction, game.player.rect , len(self.twists), game.sprite_e))


            self.gun_cooldown = max(0, self.gun_cooldown - dt)   # decreases time for bullets limitation

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

                # pushes player when he collides with the enemy
                game.player.rect.x +=  self.direction *  50 * dt

                # limits the player to stand on the screen space even being pushed.
                if game.player.rect.x < 60:
                    game.player.rect.x = 60
                if game.player.rect.x > 620:
                    game.player.rect.x = 620

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

# enemy bullets
class Bullet_e5(pygame.sprite.Sprite):



    image = pygame.image.load('evoman/images/blade.png')

    def __init__(self, location, direction, pos_p, n_twist, *groups):
        super(Bullet_e5, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction
        self.pos_p = pos_p
        self.n_twist = n_twist


    def update(self, dt, game):


        # bullets go the player's  direction marked at the shooting time
        self.rect.x += self.direction *  550 * dt
        if self.rect.bottom < self.pos_p.bottom:
            self.rect.y +=  300  * dt

        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left>736 or self.rect.bottom < 1  or self.rect.top > 512:
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
