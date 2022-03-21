from evoman.Base.SpriteDefinition import *

class Bullet_p(pygame.sprite.Sprite):
    image = pygame.image.load('evoman/images/bullet_r.png')

    def __init__(self, location, direction, n_twist, *groups):
        super(Bullet_p, self).__init__(*groups)
        self.rect = pygame.rect.Rect(location, self.image.get_size())
        self.direction = direction

        self.n_twist = n_twist

        # fits image according to the side the player is turned to
        if self.direction == 1:
            self.image = pygame.image.load('evoman/images/bullet_r.png')
        else:
            self.image = pygame.image.load('evoman/images/bullet_l.png')

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
                damage = 20 / game.level
                #game.reward += damage
                game.enemy.life = max(0, game.enemy.life - damage)

                if game.enemyn == 4:
                    # makes enemy imune to player's shooting.
                    game.enemy.imune = 1

            # removes the bullet off the screen after collision.
            self.kill()
            game.player.twists[self.n_twist] = None

            game.enemy.hurt = 5


class Player(pygame.sprite.Sprite):

    def __init__(self, location, *groups):
        super(Player, self).__init__(*groups)

        self.spriteDefinition = SpriteDefinition('evoman/images/EvoManSprites.png', 0, 0, 43, 59)
        self.updateSprite(SpriteConstants.STANDING, SpriteConstants.RIGHT)

        self.rect = pygame.rect.Rect(location, self.image.get_size())
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

    def update(self, dt, game):
        if not game.freeze_p:
            if self.inwater == 1:
                self.vx = 0.5
                self.vy = 0.5
                self.hy = -2000
            else:
                self.vx = 1
                self.vy = 1
                self.hy = -900

            left = game.actions[0]
            right = game.actions[1]
            jump = game.actions[2]
            shoot = game.actions[3]
            release = game.actions[4]

            # if the button is released before the jumping maximum height, them player stops going up.
            if release == 1 and self.resting == 0:
                self.dy = 0

            # copies last position state of the player
            last = self.rect.copy()

            # movements on the axis x (left)
            if left:

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

            # movements on the axis x (right)
            elif right:

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

            #  changes the image when player jumps
            if self.resting == 0:
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.JUMPING, SpriteConstants.RIGHT)

            new = self.rect  # copies new (after movement) position state of the player
            # controls screen walls and platforms limits agaist player
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

            # shoots, limiting time between bullets.
            if shoot == 1 and not self.gun_cooldown:

                self.shooting = 5
                self.atacked = 1  # marks if the player has atacked enemy

                # creates bullets objects according to the direction.
                if self.direction > 0:
                    self.twists.append(Bullet_p(self.rect.midright, 1, len(self.twists), game.sprite_p))

                else:
                    self.twists.append(Bullet_p(self.rect.midleft, -1, len(self.twists), game.sprite_p))

                self.gun_cooldown = 0.4  # marks time to the bullet for allowing next bullets

            else:
                self.atacked = 0

            # decreases time for limitating bullets
            self.gun_cooldown = max(0, self.gun_cooldown - dt)

            # hurt player animation
            if self.hurt > 0:
                if self.direction == -1:
                    self.updateSprite(SpriteConstants.HURTING, SpriteConstants.LEFT)
                else:
                    self.updateSprite(SpriteConstants.HURTING, SpriteConstants.RIGHT)

            self.hurt -= 1
            self.hurt = max(0, self.hurt)
            self.shooting -= 1
            self.shooting = max(0, self.shooting)

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

            # kills player in case he touches killers stuff, like spikes.
            for cell in game.tilemap.layers['triggers'].collide(self.rect, 'killers'):
                game.player.life = 0

            # focuses screen center on player
            game.tilemap.set_focus(new.x, new.y)
        else:
            game.tilemap.set_focus(self.rect.x, self.rect.y)




    def updateSprite(self, state, direction):
        self.image = self.spriteDefinition.getImage(state, direction)