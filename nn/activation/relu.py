import numpy as np

from .activation import Activation


class ReLUActivation(Activation):

    def fun(self, input):
        return np.maximum(input, 0)

    def derivative(self, input):
        return np.greater(input, 0).astype(int)
