import numpy
import struct
import binascii

# sensors for the controllers
class Sensors():


    def get(self, game):


        # calculates vertical and horizontal distances between sprites centers

        posx_p = game.player.rect.left +((game.player.rect.right - game.player.rect.left)/2)
        posy_p = game.player.rect.bottom +((game.player.rect.top - game.player.rect.bottom)/2)
        posx_e = game.enemy.rect.left +((game.enemy.rect.right - game.enemy.rect.left)/2)
        posy_e = game.enemy.rect.bottom +((game.enemy.rect.top - game.enemy.rect.bottom)/2)

        # pre-allocate values for the bullets
        param_values = [ posx_p-posx_e, posy_p-posy_e, game.player.direction, game.enemy.direction] + [0]*16

        # calculates vertical and horizontal distances between player and the center of enemy's bullets
        bullet_count = 0
        for i in range(0,len(game.enemy.twists)):
            if game.enemy.twists[i] != None:
                posx_be = game.enemy.twists[i].rect.left +((game.enemy.twists[i].rect.right - game.enemy.twists[i].rect.left)/2)
                posy_be = game.enemy.twists[i].rect.bottom +((game.enemy.twists[i].rect.top - game.enemy.twists[i].rect.bottom)/2)
                param_values[4 + bullet_count * 2] = posx_p - posx_be
                param_values[4 + bullet_count * 2 + 1] = posy_p - posy_be
                bullet_count+=1


        # applies several transformations to input variables (sensors)
        if game.inputscoded == "yes":

            types = struct.Struct('q q q q q q q q q q q q q q q q q q q q') # defines the data types of each item of the array that will be packed. (q=int, f=flo)
            packed_data = types.pack(*param_values)  # packs data as struct
            coded_variables =  binascii.hexlify(packed_data)  # converts packed data to an hexadecimal string
            coded_variables = [coded_variables[i:i+2] for i in range(0, len(coded_variables), 2)] # breaks hexadecimal string in bytes.
            coded_variables = numpy.array(map(lambda y: int(y, 16), coded_variables))  # converts bytes to integer

            param_values = coded_variables


        self.sensors = param_values # defines sensors state


        return numpy.array(self.sensors)
