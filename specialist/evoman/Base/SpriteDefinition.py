import pygame
from . import SpriteConstants

class SpriteDefinition(object):
    """Contains the properties and methods to control a SpriteSheet structure"""

    def __init__(self, fileName, origin_X, origin_Y, width, height):
        self.SpriteSheet = pygame.image.load(fileName).convert()
        self.Origin_X = origin_X
        self.Origin_Y = origin_Y
        self.Width = width
        self.Height = height

    def getImage(self, steps_X, steps_Y):
        marginX = self.Width * steps_X
        marginY = self.Height * steps_Y

        image = pygame.Surface([self.Width, self.Height]).convert()

        image.blit(self.SpriteSheet,
                   (0, 0),
                   (marginX,
                    marginY,
                    self.Width,
                    self.Height))

        image.set_colorkey(SpriteConstants.BLACK)

        return image
