import numpy as np

from .activation import Activation


class TanhActivation(Activation):

    def fun(self, input):
        return np.tanh(input)

    def derivative(self, input):
        return 1 - np.power(np.tanh(input), 2)
