################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys

from evoman.Base.SpriteDefinition import *


# player proctile
class Bullet_p(pygame.sprite.Sprite):

    def __init__(self, location, direction, n_twist, *groups, visuals):
        super(Bullet_p, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, (13, 7))
        self.direction = direction
        self.n_twist = n_twist

        if visuals:
            self.image_r = pygame.image.load('evoman/images/bullet_r.png')
            self.image_l = pygame.image.load('evoman/images/bullet_l.png')

            if self.direction == 1:
                self.image = self.image_r
            else:
                self.image = self.image_l

            self.rect = pygame.rect.Rect(location, self.image.get_size())

    def update(self, dt, game):

        # removes bullets objetcs when they transpass the screen limits
        if self.rect.right < 1 or self.rect.left > 736 or self.rect.top < 1 or self.rect.bottom > 512:
            self.kill()
            game.player.twists[self.n_twist] = None
            return

        self.rect.x += self.direction * 600 * dt  # moving on the X axis (left or tight). It adds 600*dt forward at each general game loop loop iteration, where dt controls the frames limit.

        # checks collision of player's bullet with the enemy
        if self.rect.colliderect(game.enemy.rect):

            # if enemy is not imune
            if game.enemy.imune == 0:
                # enemy loses life points, according to the difficult level of the game (the more difficult, the less it loses)
                game.enemy.life = max(0, game.enemy.life - (20 / game.level))

                if game.enemyn == 4:
                    # makes enemy imune to player's shooting.
                    game.enemy.imune = 1

            # removes the bullet off the screen after collision.
            self.kill()
            game.player.twists[self.n_twist] = None

            game.enemy.hurt = 5


# player sprite
class Player(pygame.sprite.Sprite):

    def __init__(self, location, enemyn, level, *groups, visuals):
        super(Player, self).__init__(*groups)

        self.visuals = visuals
        if visuals:
            self.spriteDefinition = SpriteDefinition('evoman/images/EvoManSprites.png', 0, 0, 43, 59)
            self.updateSprite(SpriteConstants.STANDING, SpriteConstants.RIGHT)
            self.image = self.spriteDefinition.getImage(SpriteConstants.STANDING, SpriteConstants.RIGHT)
            self.rect = pygame.rect.Rect(location, self.image.get_size())
        else:
            self.rect = pygame.rect.Rect(location, pygame.Surface([43, 59]).get_size())

        self.resting = 0
        self.dy = 0
        self.direction = 1
        self.alternate = 1
        self.gun_cooldown = 0
        self.max_life = 100
        self.life = self.max_life
        self.atacked = 0
        self.hurt = 0
        self.shooting = 0
        self.inwater = 0
        self.twists = []
        self.vx = 0
        self.vy = 0
        self.hy = 0
        self.sensors = None

    def update(self, dt, game):

        if game.freeze_p != 0 or game.start != 1:
            game.tilemap.set_focus(self.rect.x, self.rect.y)
            return
        # if the enemies are not atacking with the freezing atack (prevents player from making any movements or atacking) and also the 'start game' marker is 1.

        # checks water environment flag to regulate movements speed
        if self.inwater == 1:
            self.vx = 0.5
            self.vy = 0.5
            self.hy = -2000
        else:
            self.vx = 1
            self.vy = 1
            self.hy = -900

        jump, left, release, right, shoot = self.get_input(game)

        # if the button is released before the jumping maximum height, them player stops going up.
        if release == 1 and self.resting == 0:
            self.dy = 0

        # copies last position state of the player
        last = self.rect.copy()

        # movements on the axis x (left)
        if left:

            self.handle_left(dt)

        # movements on the axis x (right)
        elif right:

            self.handle_right(dt)


        else:
            # animation, standing up images
            if self.direction == -1:
                self.updateSprite(SpriteConstants.STANDING, SpriteConstants.LEFT)
            else:
                self.updateSprite(SpriteConstants.STANDING, SpriteConstants.RIGHT)

        # if player is touching the floor, he is allowed to jump
        if self.resting == 1 and jump == 1:
            self.dy = self.hy

        # gravity
        self.dy = min(400, self.dy + 100)
        self.rect.y += self.dy * dt * self.vy

        new = self.rect  # copies new (after movement) position state of the player

        # controls screen walls and platforms limits agaist player
        self.check_blockers(game, last, new)

        # shoots, limiting time between bullets.
        self.handle_shoot(game, shoot)

        # decreases time for limitating bullets
        self.gun_cooldown = max(0, self.gun_cooldown - dt)

        self.hurt -= 1
        self.hurt = max(0, self.hurt)
        self.shooting -= 1
        self.shooting = max(0, self.shooting)

        # kills player in case he touches killers stuff, like spikes.
        for cell in game.tilemap.layers['triggers'].collide(self.rect, 'killers'):
            game.player.life = 0

        self.update_animation(game, new)

    def get_input(self, game):
        # defines game mode for player action
        if game.playermode == 'human':  # player controlled by keyboard/joystick

            jump, left, release, right, shoot = self.human_input(game)

        elif game.playermode == 'ai':  # player controlled by AI algorithm

            jump, left, release, right, shoot = self.ai_input(game)
        return jump, left, release, right, shoot

    def update_animation(self, game, new):
        #  changes the image when player jumps
        if self.resting == 0:
            if self.direction == -1:
                self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.LEFT)
            else:
                self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.RIGHT)
        # hurt player animation
        if self.hurt > 0:
            if self.direction == -1:
                self.updateSprite(SpriteConstants.HURTING, SpriteConstants.LEFT)
            else:
                self.updateSprite(SpriteConstants.HURTING, SpriteConstants.RIGHT)
        # shooting animation
        if self.shooting > 0:
            if self.resting == 0:
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.SHOOTING_JUMPING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.SHOOTING_JUMPING, SpriteConstants.RIGHT)
            else:
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.SHOOTING, SpriteConstants.RIGHT)
        # focuses screen center on player
        if self.visuals:
            game.tilemap.set_focus(new.x, new.y)

    def handle_shoot(self, game, shoot):
        if shoot == 1 and not self.gun_cooldown:

            self.shooting = 5
            self.atacked = 1  # marks if the player has atacked enemy

            # creates bullets objects according to the direction.
            if self.direction > 0:
                self.twists.append(
                    Bullet_p(self.rect.midright, 1, len(self.twists), game.sprite_p, visuals=self.visuals))

            else:
                self.twists.append(
                    Bullet_p(self.rect.midleft, -1, len(self.twists), game.sprite_p, visuals=self.visuals))

            self.gun_cooldown = 0.4  # marks time to the bullet for allowing next bullets

            # sound effects
            if game.sound == "on" and game.playermode == "human":
                sound = pygame.mixer.Sound('evoman/sounds/scifi003.wav')
                c = pygame.mixer.Channel(2)
                c.set_volume(1)
                c.play(sound)

        else:
            self.atacked = 0

    def check_blockers(self, game, last, new):
        self.resting = 0
        for cell in game.tilemap.layers['triggers'].collide(new, 'blockers'):

            blockers = cell['blockers']

            if 'l' in blockers and last.right <= cell.left and new.right > cell.left and last.bottom > cell.top:
                new.right = cell.left

            if 'r' in blockers and last.left >= cell.right and new.left < cell.right and last.bottom > cell.top:
                new.left = cell.right

            if 't' in blockers and last.bottom <= cell.top and new.bottom > cell.top:
                self.resting = 1  # player touches the floor
                new.bottom = cell.top
                self.dy = 0

            if 'b' in blockers and last.top >= cell.bottom and new.top < cell.bottom:
                new.top = cell.bottom

    def human_input(self, game):
        # if joystick is connected, initializes it.
        if game.joy > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
        # tests if the button/key was pressed or released.
        # if the player is jumping, the release stops the jump before its maximum high is achieved
        press = 0
        release = 0
        for event in game.event:
            if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.KEYDOWN:
                press = 1
            else:
                press = 0

            if event.type == pygame.JOYBUTTONUP or event.type == pygame.KEYUP:
                release = 1
            else:
                release = 0
        # gets pressed key value
        key = pygame.key.get_pressed()
        # gets joystick value for axis x (left/right)
        left = 0
        if game.joy > 0:
            if round(joystick.get_axis(0)) == -1:
                left = 1
        if key[pygame.K_LEFT]:
            left = 1
        right = 0
        if game.joy > 0:
            if round(joystick.get_axis(0)) == 1:
                right = 1
        if key[pygame.K_RIGHT]:
            right = 1
        # gets joystick/key value for jumping
        jump = 0
        if game.joy > 0:
            if int(joystick.get_button(2)) == 1 and press == 1:
                jump = 1
        if key[pygame.K_SPACE] and press == 1:
            jump = 1
        # gets joystick/key value for shooting
        shoot = 0
        if game.joy > 0:
            if int(joystick.get_button(3)) == 1 and press == 1:
                shoot = 1
        if key[pygame.K_LSHIFT] and press == 1:
            shoot = 1
        return jump, left, release, right, shoot

    def ai_input(self, game):
        if game.time == 1:
            game.player_controller.set(game.pcont, len(self.sensors.get(game)))
        # calls the controller providing game sensors
        actions = game.player_controller.control(self.sensors.get(game), game.pcont)
        if len(actions) < 5:
            game.print_logs("ERROR: Player controller must return 5 decision variables.")
            sys.exit(0)
        left = actions[0]
        right = actions[1]
        jump = actions[2]
        shoot = actions[3]
        release = actions[4]
        return jump, left, release, right, shoot

    def handle_right(self, dt):
        self.rect.x += 200 * dt * self.vx
        self.direction = 1
        # animation, running player images alternation
        if self.alternate == 1:
            self.updateSprite(SpriteConstants.START_RUNNING, SpriteConstants.RIGHT)
        if self.alternate == 4 or self.alternate == 10:
            self.updateSprite(SpriteConstants.RUNNING_STEP1, SpriteConstants.RIGHT)
        if self.alternate == 7:
            self.updateSprite(SpriteConstants.RUNNING_STEP2, SpriteConstants.RIGHT)
        self.alternate += 1
        if self.alternate > 12:
            self.alternate = 1

    def handle_left(self, dt):
        self.rect.x -= 200 * dt * self.vx
        self.direction = -1
        # animation, running images alternation
        if self.alternate == 1:
            self.updateSprite(SpriteConstants.START_RUNNING, SpriteConstants.LEFT)
        if self.alternate == 4 or self.alternate == 10:
            self.updateSprite(SpriteConstants.RUNNING_STEP1, SpriteConstants.LEFT)
        if self.alternate == 7:
            self.updateSprite(SpriteConstants.RUNNING_STEP2, SpriteConstants.LEFT)
        self.alternate += 1
        if self.alternate > 12:
            self.alternate = 1

    def updateSprite(self, state, direction):
        if self.visuals:
            self.image = self.spriteDefinition.getImage(state, direction)
