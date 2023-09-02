################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import numpy

class Controller(object):

    def set(self, genome, n_inputs):
        pass

    def control(self, params, cont = None):

        left = numpy.random.choice([1,0])
        right = numpy.random.choice([1,0])
        jump = numpy.random.choice([1,0])
        shoot = numpy.random.choice([1,0])
        release = numpy.random.choice([1,0])

        return [left, right, jump, shoot, release]
