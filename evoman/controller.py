################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import numpy

class Controller(object):


    def control(self, params, cont = None):

        action1 = numpy.random.choice([1,0])
        action2 = numpy.random.choice([1,0])
        action3 = numpy.random.choice([1,0])
        action4 = numpy.random.choice([1,0])
        action5 = numpy.random.choice([1,0])
        action6 = numpy.random.choice([1,0])

        return [action1, action2, action3, action4, action5, action6]
